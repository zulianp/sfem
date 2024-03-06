#include "tet10_navier_stokes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "sfem_vec.h"

#include "tet10_convection.h"

static SFEM_INLINE void tet10_momentum_lhs_scalar_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t px3,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t py3,
                                                         const real_t pz0,
                                                         const real_t pz1,
                                                         const real_t pz2,
                                                         const real_t pz3,
                                                         const real_t dt,
                                                         const real_t nu,
                                                         real_t *const SFEM_RESTRICT
                                                             element_matrix) {
    // TODO
    assert(0);
}

// static SFEM_INLINE void tet10_momentum_rhs_kernel(const real_t px0,
//                                                   const real_t px1,
//                                                   const real_t px2,
//                                                   const real_t px3,
//                                                   const real_t py0,
//                                                   const real_t py1,
//                                                   const real_t py2,
//                                                   const real_t py3,
//                                                   const real_t pz0,
//                                                   const real_t pz1,
//                                                   const real_t pz2,
//                                                   const real_t pz3,
//                                                   const real_t dt,
//                                                   const real_t nu,
//                                                   real_t *const SFEM_RESTRICT u,
//                                                   real_t *const SFEM_RESTRICT element_vector) {
// 	//TODO
// 	assert(0);
// }

static SFEM_INLINE void tet4_tet10_divergence_rhs_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t px3,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t py3,
                                                         const real_t pz0,
                                                         const real_t pz1,
                                                         const real_t pz2,
                                                         const real_t pz3,
                                                         const real_t dt,
                                                         const real_t rho,
                                                         const real_t *const SFEM_RESTRICT u,
                                                         real_t *const SFEM_RESTRICT
                                                             element_vector) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = -px0 + px3;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = -py0 + py2;
    const real_t x8 = -py0 + py3;
    const real_t x9 = -px0 + px2;
    const real_t x10 = -py0 + py1;
    const real_t x11 = -pz0 + pz2;
    const real_t x12 = x0 * x11;
    const real_t x13 = x1 * x9;
    const real_t x14 = x10 * x11 * x3 - x10 * x13 - x12 * x8 + x2 * x7 + x4 * x8 * x9 - x5 * x7;
    const real_t x15 = 1.0 / x14;
    const real_t x16 = x11 * x3 - x13;
    const real_t x17 = -x12 + x4 * x9;
    const real_t x18 = -x3 * x7 + x8 * x9;
    const real_t x19 = -x0 * x8 + x10 * x3;
    const real_t x20 = x0 * x7 - x10 * x9;
    const real_t x21 = x1 * x7 - x11 * x8;
    const real_t x22 = -x1 * x10 + x4 * x8;
    const real_t x23 = x10 * x11 - x4 * x7;
    const real_t x24 = (1.0 / 30.0) * x15;
    const real_t x25 = u[14] * x24;
    const real_t x26 = x25 * x6;
    const real_t x27 = x17 * x25;
    const real_t x28 = u[16] * x24;
    const real_t x29 = x17 * x28;
    const real_t x30 = x16 * x28;
    const real_t x31 = u[17] * x24;
    const real_t x32 = x31 * x6;
    const real_t x33 = x16 * x31;
    const real_t x34 = u[24] * x24;
    const real_t x35 = x20 * x34;
    const real_t x36 = x19 * x34;
    const real_t x37 = u[26] * x24;
    const real_t x38 = x20 * x37;
    const real_t x39 = x18 * x37;
    const real_t x40 = u[27] * x24;
    const real_t x41 = x18 * x40;
    const real_t x42 = x19 * x40;
    const real_t x43 = u[4] * x24;
    const real_t x44 = x23 * x43;
    const real_t x45 = x22 * x43;
    const real_t x46 = u[6] * x24;
    const real_t x47 = x23 * x46;
    const real_t x48 = x21 * x46;
    const real_t x49 = u[7] * x24;
    const real_t x50 = x21 * x49;
    const real_t x51 = x22 * x49;
    const real_t x52 = (1.0 / 40.0) * x15;
    const real_t x53 = u[0] * x52;
    const real_t x54 = u[10] * x52;
    const real_t x55 = u[20] * x52;
    const real_t x56 = (1.0 / 120.0) * x15;
    const real_t x57 = u[11] * x16;
    const real_t x58 = x56 * x57;
    const real_t x59 = x56 * x6;
    const real_t x60 = u[12] * x59;
    const real_t x61 = x17 * x56;
    const real_t x62 = u[13] * x61;
    const real_t x63 = u[1] * x21;
    const real_t x64 = x56 * x63;
    const real_t x65 = u[21] * x18;
    const real_t x66 = x56 * x65;
    const real_t x67 = x19 * x56;
    const real_t x68 = u[22] * x67;
    const real_t x69 = x20 * x56;
    const real_t x70 = u[23] * x69;
    const real_t x71 = x22 * x56;
    const real_t x72 = u[2] * x71;
    const real_t x73 = x23 * x56;
    const real_t x74 = u[3] * x73;
    const real_t x75 = -x16 * x25 - x18 * x34 - x21 * x43;
    const real_t x76 = -x19 * x37 - x22 * x46 - x28 * x6;
    const real_t x77 = -x17 * x31 - x20 * x40 - x23 * x49;
    const real_t x78 = rho * x14 / dt;
    const real_t x79 = (1.0 / 15.0) * x15;
    const real_t x80 = u[14] * x79;
    const real_t x81 = u[24] * x79;
    const real_t x82 = u[4] * x79;
    const real_t x83 = u[15] * x79;
    const real_t x84 = x17 * x79;
    const real_t x85 = u[25] * x79;
    const real_t x86 = x20 * x79;
    const real_t x87 = u[5] * x79;
    const real_t x88 = x23 * x79;
    const real_t x89 = u[18] * x24;
    const real_t x90 = u[19] * x24;
    const real_t x91 = u[28] * x24;
    const real_t x92 = u[29] * x24;
    const real_t x93 = u[8] * x24;
    const real_t x94 = u[9] * x24;
    const real_t x95 = u[0] * x73;
    const real_t x96 = u[0] * x21 * x56;
    const real_t x97 = u[0] * x71;
    const real_t x98 = u[10] * x59;
    const real_t x99 = u[10] * x61;
    const real_t x100 = u[10] * x16 * x56;
    const real_t x101 = u[20] * x69;
    const real_t x102 = u[20] * x18 * x56;
    const real_t x103 = u[20] * x67;
    const real_t x104 = x100 + x101 + x102 + x103 + x16 * x89 + x18 * x91 + x19 * x92 + x21 * x93 +
                        x22 * x94 - x32 - x33 - x41 - x42 - x50 - x51 + x6 * x90 - x62 - x70 - x74 +
                        x95 + x96 + x97 + x98 + x99;
    const real_t x105 = u[15] * x24;
    const real_t x106 = u[25] * x24;
    const real_t x107 = u[5] * x24;
    const real_t x108 = x105 * x16 + x106 * x18 + x107 * x21 + x17 * x90 + x20 * x92 + x23 * x94 -
                        x29 - x30 - x38 - x39 - x47 - x48 - x60 - x68 - x72;
    const real_t x109 = x16 * x79;
    const real_t x110 = x18 * x79;
    const real_t x111 = x21 * x79;
    const real_t x112 = x105 * x6 + x106 * x19 + x107 * x22 + x17 * x89 + x20 * x91 + x23 * x93 -
                        x26 - x27 - x35 - x36 - x44 - x45 - x58 - x64 - x66;
    const real_t x113 = x6 * x79;
    const real_t x114 = x19 * x79;
    const real_t x115 = x22 * x79;
    element_vector[0] =
        -x78 *
        ((1.0 / 30.0) * u[15] * x15 * x16 + (1.0 / 30.0) * u[15] * x15 * x6 +
         (1.0 / 30.0) * u[18] * x15 * x16 + (1.0 / 30.0) * u[18] * x15 * x17 +
         (1.0 / 30.0) * u[19] * x15 * x17 + (1.0 / 30.0) * u[19] * x15 * x6 +
         (1.0 / 30.0) * u[25] * x15 * x18 + (1.0 / 30.0) * u[25] * x15 * x19 +
         (1.0 / 30.0) * u[28] * x15 * x18 + (1.0 / 30.0) * u[28] * x15 * x20 +
         (1.0 / 30.0) * u[29] * x15 * x19 + (1.0 / 30.0) * u[29] * x15 * x20 +
         (1.0 / 30.0) * u[5] * x15 * x21 + (1.0 / 30.0) * u[5] * x15 * x22 +
         (1.0 / 30.0) * u[8] * x15 * x21 + (1.0 / 30.0) * u[8] * x15 * x23 +
         (1.0 / 30.0) * u[9] * x15 * x22 + (1.0 / 30.0) * u[9] * x15 * x23 - x16 * x54 - x17 * x54 -
         x18 * x55 - x19 * x55 - x20 * x55 - x21 * x53 - x22 * x53 - x23 * x53 - x26 - x27 - x29 -
         x30 - x32 - x33 - x35 - x36 - x38 - x39 - x41 - x42 - x44 - x45 - x47 - x48 - x50 - x51 -
         x54 * x6 - x58 - x60 - x62 - x64 - x66 - x68 - x70 - x72 - x74 - x75 - x76 - x77);
    element_vector[1] =
        -x78 * (u[18] * x84 + u[28] * x86 + u[8] * x88 + x104 + x108 - x17 * x80 - x19 * x81 +
                x19 * x85 - x20 * x81 - x22 * x82 + x22 * x87 - x23 * x82 + x52 * x57 + x52 * x63 +
                x52 * x65 - x6 * x80 + x6 * x83 + x75);
    element_vector[2] =
        -x78 * (u[12] * x52 * x6 - u[16] * x109 - u[16] * x84 + u[19] * x84 + u[22] * x19 * x52 -
                u[26] * x110 - u[26] * x86 + u[29] * x86 + u[2] * x22 * x52 - u[6] * x111 -
                u[6] * x88 + u[9] * x88 + x104 + x112 + x16 * x83 + x18 * x85 + x21 * x87 + x76);
    element_vector[3] =
        -x78 * (u[13] * x17 * x52 - u[17] * x109 - u[17] * x113 + u[18] * x109 + u[19] * x113 +
                u[23] * x20 * x52 - u[27] * x110 - u[27] * x114 + u[28] * x110 + u[29] * x114 +
                u[3] * x23 * x52 - u[7] * x111 - u[7] * x115 + u[8] * x111 + u[9] * x115 + x100 +
                x101 + x102 + x103 + x108 + x112 + x77 + x95 + x96 + x97 + x98 + x99);
}

static SFEM_INLINE void tet10_tet4_rhs_correction_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t px3,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t py3,
                                                         const real_t pz0,
                                                         const real_t pz1,
                                                         const real_t pz2,
                                                         const real_t pz3,
                                                         const real_t dt,
                                                         const real_t rho,
                                                         const real_t *const SFEM_RESTRICT p,
                                                         real_t *const SFEM_RESTRICT
                                                             element_vector) {
    const real_t x0 = -py0 + py1;
    const real_t x1 = -pz0 + pz2;
    const real_t x2 = x0 * x1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = -px0 + px1;
    const real_t x8 = -pz0 + pz3;
    const real_t x9 = x3 * x8;
    const real_t x10 = -px0 + px2;
    const real_t x11 = -py0 + py3;
    const real_t x12 = -px0 + px3;
    const real_t x13 = x1 * x11;
    const real_t x14 = x0 * x8;
    const real_t x15 = x10 * x11 * x4 - x10 * x14 + x12 * x2 - x12 * x5 - x13 * x7 + x7 * x9;
    const real_t x16 = 1.0 / x15;
    const real_t x17 = (1.0 / 120.0) * x16;
    const real_t x18 = x17 * x6;
    const real_t x19 = x11 * x4 - x14;
    const real_t x20 = x17 * x19;
    const real_t x21 = -x13 + x9;
    const real_t x22 = x17 * x21;
    const real_t x23 = dt * x15 / rho;
    const real_t x24 =
        -x23 * (p[0] * x18 + p[0] * x20 + p[0] * x22 - p[1] * x22 - p[2] * x20 - p[3] * x18);
    const real_t x25 = (1.0 / 30.0) * p[0] * x16;
    const real_t x26 = -x23 * ((1.0 / 30.0) * p[1] * x16 * x21 + (1.0 / 30.0) * p[2] * x16 * x19 +
                               (1.0 / 30.0) * p[3] * x16 * x6 - x19 * x25 - x21 * x25 - x25 * x6);
    const real_t x27 = -x1 * x7 + x10 * x4;
    const real_t x28 = x17 * x27;
    const real_t x29 = -x12 * x4 + x7 * x8;
    const real_t x30 = x17 * x29;
    const real_t x31 = x1 * x12 - x10 * x8;
    const real_t x32 = x17 * x31;
    const real_t x33 =
        -x23 * (p[0] * x28 + p[0] * x30 + p[0] * x32 - p[1] * x32 - p[2] * x30 - p[3] * x28);
    const real_t x34 = -x23 * ((1.0 / 30.0) * p[1] * x16 * x31 + (1.0 / 30.0) * p[2] * x16 * x29 +
                               (1.0 / 30.0) * p[3] * x16 * x27 - x25 * x27 - x25 * x29 - x25 * x31);
    const real_t x35 = -x0 * x10 + x3 * x7;
    const real_t x36 = x17 * x35;
    const real_t x37 = x0 * x12 - x11 * x7;
    const real_t x38 = x17 * x37;
    const real_t x39 = x10 * x11 - x12 * x3;
    const real_t x40 = x17 * x39;
    const real_t x41 =
        -x23 * (p[0] * x36 + p[0] * x38 + p[0] * x40 - p[1] * x40 - p[2] * x38 - p[3] * x36);
    const real_t x42 = -x23 * ((1.0 / 30.0) * p[1] * x16 * x39 + (1.0 / 30.0) * p[2] * x16 * x37 +
                               (1.0 / 30.0) * p[3] * x16 * x35 - x25 * x35 - x25 * x37 - x25 * x39);
    element_vector[0] = x24;
    element_vector[1] = x24;
    element_vector[2] = x24;
    element_vector[3] = x24;
    element_vector[4] = x26;
    element_vector[5] = x26;
    element_vector[6] = x26;
    element_vector[7] = x26;
    element_vector[8] = x26;
    element_vector[9] = x26;
    element_vector[10] = x33;
    element_vector[11] = x33;
    element_vector[12] = x33;
    element_vector[13] = x33;
    element_vector[14] = x34;
    element_vector[15] = x34;
    element_vector[16] = x34;
    element_vector[17] = x34;
    element_vector[18] = x34;
    element_vector[19] = x34;
    element_vector[20] = x41;
    element_vector[21] = x41;
    element_vector[22] = x41;
    element_vector[23] = x41;
    element_vector[24] = x42;
    element_vector[25] = x42;
    element_vector[26] = x42;
    element_vector[27] = x42;
    element_vector[28] = x42;
    element_vector[29] = x42;
}

static SFEM_INLINE void tet10_add_diffusion_rhs_kernel(const real_t px0,
                                                       const real_t px1,
                                                       const real_t px2,
                                                       const real_t px3,
                                                       const real_t py0,
                                                       const real_t py1,
                                                       const real_t py2,
                                                       const real_t py3,
                                                       const real_t pz0,
                                                       const real_t pz1,
                                                       const real_t pz2,
                                                       const real_t pz3,
                                                       const real_t dt,
                                                       const real_t nu,
                                                       real_t *const SFEM_RESTRICT u,
                                                       real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py2;
    const real_t x3 = x1 * x2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -px0 + px2;
    const real_t x6 = -py0 + py3;
    const real_t x7 = x5 * x6;
    const real_t x8 = -pz0 + pz2;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x1 * x6;
    const real_t x12 = x10 * x5;
    const real_t x13 = x2 * x9;
    const real_t x14 = -x0 * x12 + x0 * x3 + x10 * x8 * x9 - x11 * x8 - x13 * x4 + x4 * x7;
    const real_t x15 = pow(x14, -2);
    const real_t x16 = (1.0 / 10.0) * x15;
    const real_t x17 = -x12 + x3;
    const real_t x18 = pow(x17, 2);
    const real_t x19 = u[0] * x18;
    const real_t x20 = x10 * x9 - x11;
    const real_t x21 = pow(x20, 2);
    const real_t x22 = u[0] * x16;
    const real_t x23 = -x13 + x7;
    const real_t x24 = pow(x23, 2);
    const real_t x25 = (1.0 / 30.0) * x15;
    const real_t x26 = x24 * x25;
    const real_t x27 = u[1] * x26;
    const real_t x28 = x21 * x25;
    const real_t x29 = u[2] * x28;
    const real_t x30 = x18 * x25;
    const real_t x31 = u[3] * x30;
    const real_t x32 = u[4] * x25;
    const real_t x33 = x18 * x32;
    const real_t x34 = x21 * x32;
    const real_t x35 = (2.0 / 15.0) * x15;
    const real_t x36 = u[4] * x35;
    const real_t x37 = -x24 * x36;
    const real_t x38 = u[5] * x28;
    const real_t x39 = u[5] * x26;
    const real_t x40 = u[6] * x25;
    const real_t x41 = x18 * x40;
    const real_t x42 = u[6] * x35;
    const real_t x43 = -x21 * x42;
    const real_t x44 = x24 * x40;
    const real_t x45 = u[7] * x35;
    const real_t x46 = -x18 * x45;
    const real_t x47 = u[7] * x25;
    const real_t x48 = x21 * x47;
    const real_t x49 = x24 * x47;
    const real_t x50 = u[8] * x30;
    const real_t x51 = u[8] * x26;
    const real_t x52 = u[9] * x30;
    const real_t x53 = u[9] * x28;
    const real_t x54 = (1.0 / 5.0) * u[0];
    const real_t x55 = x15 * x20;
    const real_t x56 = x17 * x55;
    const real_t x57 = x15 * x23;
    const real_t x58 = x17 * x57;
    const real_t x59 = x20 * x57;
    const real_t x60 = x17 * x25;
    const real_t x61 = u[1] * x23;
    const real_t x62 = x60 * x61;
    const real_t x63 = x20 * x25;
    const real_t x64 = x61 * x63;
    const real_t x65 = x20 * x60;
    const real_t x66 = u[2] * x65;
    const real_t x67 = x23 * x63;
    const real_t x68 = u[2] * x67;
    const real_t x69 = u[3] * x65;
    const real_t x70 = x23 * x60;
    const real_t x71 = u[3] * x70;
    const real_t x72 = (1.0 / 15.0) * x17;
    const real_t x73 = x55 * x72;
    const real_t x74 = (1.0 / 6.0) * x17;
    const real_t x75 = u[4] * x57;
    const real_t x76 = (1.0 / 6.0) * x20;
    const real_t x77 = u[5] * x65;
    const real_t x78 = u[5] * x70;
    const real_t x79 = (1.0 / 15.0) * x20;
    const real_t x80 = x57 * x79;
    const real_t x81 = x55 * x74;
    const real_t x82 = u[6] * x57;
    const real_t x83 = u[7] * x57;
    const real_t x84 = u[8] * x65;
    const real_t x85 = x57 * x72;
    const real_t x86 = u[8] * x67;
    const real_t x87 = u[9] * x70;
    const real_t x88 = u[9] * x67;
    const real_t x89 = dt * nu * x14;
    const real_t x90 = -x1 * x8 + x4 * x5;
    const real_t x91 = pow(x90, 2);
    const real_t x92 = x0 * x1 - x4 * x9;
    const real_t x93 = pow(x92, 2);
    const real_t x94 = -x0 * x5 + x8 * x9;
    const real_t x95 = pow(x94, 2);
    const real_t x96 = x25 * x95;
    const real_t x97 = u[1] * x96;
    const real_t x98 = x25 * x93;
    const real_t x99 = u[2] * x98;
    const real_t x100 = x25 * x91;
    const real_t x101 = u[3] * x100;
    const real_t x102 = x32 * x91;
    const real_t x103 = x32 * x93;
    const real_t x104 = -x36 * x95;
    const real_t x105 = u[5] * x98;
    const real_t x106 = u[5] * x96;
    const real_t x107 = x40 * x91;
    const real_t x108 = -x42 * x93;
    const real_t x109 = x40 * x95;
    const real_t x110 = -x45 * x91;
    const real_t x111 = x47 * x93;
    const real_t x112 = x47 * x95;
    const real_t x113 = u[8] * x100;
    const real_t x114 = u[8] * x96;
    const real_t x115 = u[9] * x100;
    const real_t x116 = u[9] * x98;
    const real_t x117 = x15 * x90;
    const real_t x118 = x54 * x92;
    const real_t x119 = x15 * x94;
    const real_t x120 = x119 * x90;
    const real_t x121 = u[1] * x94;
    const real_t x122 = x25 * x90;
    const real_t x123 = x121 * x122;
    const real_t x124 = x25 * x92;
    const real_t x125 = x121 * x124;
    const real_t x126 = u[2] * x124;
    const real_t x127 = x126 * x90;
    const real_t x128 = x126 * x94;
    const real_t x129 = x124 * x90;
    const real_t x130 = u[3] * x129;
    const real_t x131 = x122 * x94;
    const real_t x132 = u[3] * x131;
    const real_t x133 = (1.0 / 15.0) * x92;
    const real_t x134 = x117 * x133;
    const real_t x135 = u[4] * x119;
    const real_t x136 = (1.0 / 6.0) * x90;
    const real_t x137 = (1.0 / 6.0) * x92;
    const real_t x138 = u[5] * x129;
    const real_t x139 = u[5] * x131;
    const real_t x140 = x119 * x133;
    const real_t x141 = u[6] * x137;
    const real_t x142 = (1.0 / 15.0) * u[6];
    const real_t x143 = x117 * x137;
    const real_t x144 = u[7] * x119;
    const real_t x145 = u[8] * x129;
    const real_t x146 = (1.0 / 15.0) * u[8];
    const real_t x147 = x124 * x94;
    const real_t x148 = u[8] * x147;
    const real_t x149 = u[9] * x131;
    const real_t x150 = u[9] * x147;
    const real_t x151 = x10 * x8 - x2 * x4;
    const real_t x152 = pow(x151, 2);
    const real_t x153 = -x0 * x10 + x4 * x6;
    const real_t x154 = pow(x153, 2);
    const real_t x155 = x0 * x2 - x6 * x8;
    const real_t x156 = pow(x155, 2);
    const real_t x157 = x156 * x25;
    const real_t x158 = u[1] * x157;
    const real_t x159 = x154 * x25;
    const real_t x160 = u[2] * x159;
    const real_t x161 = x152 * x25;
    const real_t x162 = u[3] * x161;
    const real_t x163 = x152 * x32;
    const real_t x164 = x154 * x32;
    const real_t x165 = -x156 * x36;
    const real_t x166 = u[5] * x159;
    const real_t x167 = u[5] * x157;
    const real_t x168 = x152 * x40;
    const real_t x169 = -x154 * x42;
    const real_t x170 = x156 * x40;
    const real_t x171 = -x152 * x45;
    const real_t x172 = x154 * x47;
    const real_t x173 = x156 * x47;
    const real_t x174 = u[8] * x161;
    const real_t x175 = u[8] * x157;
    const real_t x176 = u[9] * x161;
    const real_t x177 = u[9] * x159;
    const real_t x178 = x15 * x153;
    const real_t x179 = x151 * x178;
    const real_t x180 = x15 * x155;
    const real_t x181 = x151 * x180;
    const real_t x182 = x153 * x180;
    const real_t x183 = x151 * x25;
    const real_t x184 = u[1] * x155;
    const real_t x185 = x183 * x184;
    const real_t x186 = x153 * x25;
    const real_t x187 = x184 * x186;
    const real_t x188 = x153 * x183;
    const real_t x189 = u[2] * x188;
    const real_t x190 = x155 * x186;
    const real_t x191 = u[2] * x190;
    const real_t x192 = u[3] * x188;
    const real_t x193 = x155 * x183;
    const real_t x194 = u[3] * x193;
    const real_t x195 = (1.0 / 15.0) * x179;
    const real_t x196 = (1.0 / 6.0) * x151;
    const real_t x197 = u[4] * x180;
    const real_t x198 = (1.0 / 6.0) * x153;
    const real_t x199 = u[5] * x188;
    const real_t x200 = u[5] * x193;
    const real_t x201 = (1.0 / 15.0) * x153;
    const real_t x202 = x180 * x201;
    const real_t x203 = x178 * x196;
    const real_t x204 = x180 * x198;
    const real_t x205 = u[7] * x180;
    const real_t x206 = u[8] * x188;
    const real_t x207 = u[8] * x190;
    const real_t x208 = u[9] * x193;
    const real_t x209 = u[9] * x190;
    const real_t x210 = u[1] * x16;
    const real_t x211 = x16 * x23;
    const real_t x212 = u[4] * x211;
    const real_t x213 = -x68;
    const real_t x214 = u[0] * x70;
    const real_t x215 = u[0] * x67;
    const real_t x216 = x214 + x215;
    const real_t x217 = u[0] * x26;
    const real_t x218 = x217 - x71;
    const real_t x219 = x20 * x211;
    const real_t x220 = x20 * x23;
    const real_t x221 = u[5] * x219 + x220 * x47;
    const real_t x222 = x17 * x211;
    const real_t x223 = x17 * x23;
    const real_t x224 = u[8] * x222 + x223 * x40;
    const real_t x225 = x16 * x94;
    const real_t x226 = x225 * x90;
    const real_t x227 = -x128;
    const real_t x228 = u[0] * x147;
    const real_t x229 = u[0] * x131;
    const real_t x230 = x228 + x229;
    const real_t x231 = u[0] * x96;
    const real_t x232 = -x132 + x231;
    const real_t x233 = x225 * x92;
    const real_t x234 = x92 * x94;
    const real_t x235 = u[5] * x233 + x234 * x47;
    const real_t x236 = x90 * x94;
    const real_t x237 = u[8] * x226 + x236 * x40;
    const real_t x238 = x155 * x16;
    const real_t x239 = u[4] * x238;
    const real_t x240 = -x191;
    const real_t x241 = u[0] * x193;
    const real_t x242 = u[0] * x190;
    const real_t x243 = x241 + x242;
    const real_t x244 = u[0] * x157;
    const real_t x245 = -x194 + x244;
    const real_t x246 = x153 * x238;
    const real_t x247 = x153 * x155;
    const real_t x248 = u[5] * x246 + x247 * x47;
    const real_t x249 = x151 * x238;
    const real_t x250 = x151 * x155;
    const real_t x251 = u[8] * x249 + x250 * x40;
    const real_t x252 = u[2] * x16;
    const real_t x253 = u[6] * x16;
    const real_t x254 = x17 * x20;
    const real_t x255 = -x69;
    const real_t x256 = u[0] * x65;
    const real_t x257 = x215 + x256;
    const real_t x258 = u[0] * x28;
    const real_t x259 = x258 - x64;
    const real_t x260 = u[9] * x16;
    const real_t x261 = x254 * x260 + x254 * x32;
    const real_t x262 = x90 * x92;
    const real_t x263 = -x130;
    const real_t x264 = u[0] * x129;
    const real_t x265 = x228 + x264;
    const real_t x266 = u[0] * x98;
    const real_t x267 = -x125 + x266;
    const real_t x268 = x260 * x262 + x262 * x32;
    const real_t x269 = x151 * x153;
    const real_t x270 = -x192;
    const real_t x271 = u[0] * x188;
    const real_t x272 = x242 + x271;
    const real_t x273 = u[0] * x159;
    const real_t x274 = -x187 + x273;
    const real_t x275 = x260 * x269 + x269 * x32;
    const real_t x276 = u[3] * x16;
    const real_t x277 = u[7] * x16;
    const real_t x278 = -x62;
    const real_t x279 = x214 + x256;
    const real_t x280 = x19 * x25;
    const real_t x281 = x280 - x66;
    const real_t x282 = -x123;
    const real_t x283 = x229 + x264;
    const real_t x284 = u[0] * x100;
    const real_t x285 = -x127 + x284;
    const real_t x286 = -x185;
    const real_t x287 = x241 + x271;
    const real_t x288 = u[0] * x161;
    const real_t x289 = -x189 + x288;
    const real_t x290 = (4.0 / 15.0) * x15;
    const real_t x291 = u[4] * x290;
    const real_t x292 = -x220 * x291;
    const real_t x293 = u[6] * x290;
    const real_t x294 = -x220 * x293;
    const real_t x295 = x24 * x35;
    const real_t x296 = u[9] * x35;
    const real_t x297 = u[5] * x290;
    const real_t x298 = u[8] * x35;
    const real_t x299 = u[8] * x290;
    const real_t x300 = -x21 * x291 + x21 * x296 + x21 * x297 - x21 * x45 + x210 * x220 +
                        x220 * x298 + x254 * x299 - x254 * x42 + x255 + x258;
    const real_t x301 = -x223 * x291;
    const real_t x302 = u[7] * x290;
    const real_t x303 = -x223 * x302;
    const real_t x304 = u[5] * x35;
    const real_t x305 = -x18 * x291 + x18 * x296 + x18 * x299 - x18 * x42 + x210 * x223 +
                        x223 * x304 + x254 * x297 - x254 * x45 + x281 + x301 + x303;
    const real_t x306 = -x29;
    const real_t x307 = u[9] * x290;
    const real_t x308 = x254 * x307;
    const real_t x309 = u[0] * x73 + x306 + x308;
    const real_t x310 = x223 * x299;
    const real_t x311 = u[0] * x57;
    const real_t x312 = -2.0 / 15.0 * u[6] * x15 * x17 * x23 + x310 + x311 * x74;
    const real_t x313 = x220 * x297;
    const real_t x314 = -x31;
    const real_t x315 = -2.0 / 15.0 * u[7] * x15 * x20 * x23 + x311 * x76 + x313 + x314;
    const real_t x316 = -x234 * x291;
    const real_t x317 = -x234 * x293;
    const real_t x318 = x35 * x95;
    const real_t x319 = u[5] * x93;
    const real_t x320 = x210 * x234 + x234 * x298 + x262 * x299 - x262 * x42 + x263 + x266 +
                        x290 * x319 - x291 * x93 + x296 * x93 - x45 * x93;
    const real_t x321 = x290 * x91;
    const real_t x322 = -x236 * x291;
    const real_t x323 = -x236 * x302;
    const real_t x324 = -u[4] * x321 + u[8] * x321 + x210 * x236 + x236 * x304 + x262 * x297 -
                        x262 * x45 + x285 + x296 * x91 + x322 + x323 - x42 * x91;
    const real_t x325 = -x99;
    const real_t x326 = x262 * x307;
    const real_t x327 = u[0] * x134 + x325 + x326;
    const real_t x328 = x236 * x299;
    const real_t x329 = u[0] * x119;
    const real_t x330 = -2.0 / 15.0 * u[6] * x15 * x90 * x94 + x136 * x329 + x328;
    const real_t x331 = x234 * x297;
    const real_t x332 = -x101;
    const real_t x333 = -2.0 / 15.0 * u[7] * x15 * x92 * x94 + x137 * x329 + x331 + x332;
    const real_t x334 = -x247 * x291;
    const real_t x335 = -x247 * x293;
    const real_t x336 = x156 * x35;
    const real_t x337 = -x154 * x291 + x154 * x296 + x154 * x297 - x154 * x45 + x210 * x247 +
                        x247 * x298 + x269 * x299 - x269 * x42 + x270 + x273;
    const real_t x338 = -x250 * x291;
    const real_t x339 = -x250 * x302;
    const real_t x340 = -x152 * x291 + x152 * x296 + x152 * x299 - x152 * x42 + x210 * x250 +
                        x250 * x304 + x269 * x297 - x269 * x45 + x289 + x338 + x339;
    const real_t x341 = -x160;
    const real_t x342 = x269 * x307;
    const real_t x343 = u[0] * x195 + x341 + x342;
    const real_t x344 = x250 * x299;
    const real_t x345 = x180 * x196;
    const real_t x346 = u[0] * x345 - 2.0 / 15.0 * u[6] * x15 * x151 * x155 + x344;
    const real_t x347 = x247 * x297;
    const real_t x348 = -x162;
    const real_t x349 = u[0] * x204 - 2.0 / 15.0 * u[7] * x15 * x153 * x155 + x347 + x348;
    const real_t x350 = x223 * x298;
    const real_t x351 = -x27;
    const real_t x352 = u[0] * x80 + x306 + x313 + x351;
    const real_t x353 = -x254 * x291;
    const real_t x354 = -x223 * x293;
    const real_t x355 = -x220 * x302;
    const real_t x356 = x254 * x296 + x353 + x354 + x355;
    const real_t x357 = u[8] * x295 + x218 + x220 * x252 + x220 * x296 + x223 * x307 - x223 * x36 -
                        x24 * x293 + x24 * x297 - x24 * x45 + x292 + x294;
    const real_t x358 = x236 * x298;
    const real_t x359 = -x97;
    const real_t x360 = u[0] * x140 + x325 + x331 + x359;
    const real_t x361 = -x262 * x291;
    const real_t x362 = -x236 * x293;
    const real_t x363 = -x234 * x302;
    const real_t x364 = x262 * x296 + x361 + x362 + x363;
    const real_t x365 = u[8] * x318 + x232 + x234 * x252 + x234 * x296 + x236 * x307 - x236 * x36 -
                        x293 * x95 + x297 * x95 + x316 + x317 - x45 * x95;
    const real_t x366 = x250 * x298;
    const real_t x367 = -x158;
    const real_t x368 = u[0] * x202 + x341 + x347 + x367;
    const real_t x369 = -x269 * x291;
    const real_t x370 = -x250 * x293;
    const real_t x371 = -x247 * x302;
    const real_t x372 = x269 * x296 + x369 + x370 + x371;
    const real_t x373 = u[8] * x336 - x156 * x293 + x156 * x297 - x156 * x45 + x245 + x247 * x252 +
                        x247 * x296 + x250 * x307 - x250 * x36 + x334 + x335;
    const real_t x374 = -x254 * x293;
    const real_t x375 = -x254 * x302;
    const real_t x376 = x21 * x35;
    const real_t x377 = u[0] * x85 + x310 + x351;
    const real_t x378 = -x18 * x293 + x18 * x298 + x18 * x307 - x18 * x36 + x223 * x297 -
                        x223 * x45 + x252 * x254 + x254 * x304 + x278 + x280;
    const real_t x379 = u[0] * x81 - 2.0 / 15.0 * u[4] * x15 * x17 * x20 + x308;
    const real_t x380 = -x262 * x293;
    const real_t x381 = -x262 * x302;
    const real_t x382 = x35 * x93;
    const real_t x383 = (1.0 / 15.0) * u[0];
    const real_t x384 = x120 * x383 + x328 + x359;
    const real_t x385 = -u[6] * x321 + u[9] * x321 + x236 * x297 - x236 * x45 + x252 * x262 +
                        x262 * x304 + x282 + x284 + x298 * x91 - x36 * x91;
    const real_t x386 = u[0] * x143 - 2.0 / 15.0 * u[4] * x15 * x90 * x92 + x326;
    const real_t x387 = -x269 * x293;
    const real_t x388 = -x269 * x302;
    const real_t x389 = x154 * x35;
    const real_t x390 = x181 * x383 + x344 + x367;
    const real_t x391 = -x152 * x293 + x152 * x298 + x152 * x307 - x152 * x36 + x250 * x297 -
                        x250 * x45 + x252 * x269 + x269 * x304 + x286 + x288;
    const real_t x392 = u[0] * x203 - 2.0 / 15.0 * u[4] * x15 * x151 * x153 + x342;
    const real_t x393 = u[3] * x35;
    const real_t x394 = u[5] * x295 + x213 + x217 + x220 * x307 - x220 * x36 + x223 * x276 +
                        x223 * x296 + x24 * x299 - x24 * x302 - x24 * x42;
    const real_t x395 = u[5] * x376 - x21 * x302 + x21 * x307 - x21 * x36 + x220 * x299 -
                        x220 * x42 + x254 * x276 + x254 * x298 + x259 + x374 + x375;
    const real_t x396 = u[0] * x35;
    const real_t x397 = u[5] * x318 + x227 + x231 + x234 * x307 - x234 * x36 + x236 * x276 +
                        x236 * x296 + x299 * x95 - x302 * x95 - x42 * x95;
    const real_t x398 = x234 * x299 - x234 * x42 + x262 * x276 + x262 * x298 + x267 - x302 * x93 +
                        x307 * x93 + x319 * x35 - x36 * x93 + x380 + x381;
    const real_t x399 = u[5] * x336 + x156 * x299 - x156 * x302 - x156 * x42 + x240 + x244 +
                        x247 * x307 - x247 * x36 + x250 * x276 + x250 * x296;
    const real_t x400 = u[5] * x389 - x154 * x302 + x154 * x307 - x154 * x36 + x247 * x299 -
                        x247 * x42 + x269 * x276 + x269 * x298 + x274 + x387 + x388;
    const real_t x401 = x220 * x304 + x314;
    const real_t x402 = x234 * x304 + x332;
    const real_t x403 = x247 * x304 + x348;
    const real_t x404 = u[10] * x16;
    const real_t x405 = u[11] * x26;
    const real_t x406 = u[12] * x28;
    const real_t x407 = u[13] * x30;
    const real_t x408 = u[14] * x30;
    const real_t x409 = u[14] * x28;
    const real_t x410 = -u[14] * x295;
    const real_t x411 = u[15] * x28;
    const real_t x412 = u[15] * x26;
    const real_t x413 = u[16] * x30;
    const real_t x414 = -u[16] * x376;
    const real_t x415 = u[16] * x26;
    const real_t x416 = u[17] * x35;
    const real_t x417 = -x18 * x416;
    const real_t x418 = u[17] * x28;
    const real_t x419 = u[17] * x26;
    const real_t x420 = u[18] * x30;
    const real_t x421 = u[18] * x26;
    const real_t x422 = u[19] * x30;
    const real_t x423 = u[19] * x28;
    const real_t x424 = (1.0 / 5.0) * u[10];
    const real_t x425 = u[11] * x70;
    const real_t x426 = u[11] * x67;
    const real_t x427 = u[12] * x65;
    const real_t x428 = u[12] * x67;
    const real_t x429 = u[13] * x65;
    const real_t x430 = u[13] * x70;
    const real_t x431 = u[14] * x57;
    const real_t x432 = u[15] * x65;
    const real_t x433 = u[15] * x70;
    const real_t x434 = x57 * x76;
    const real_t x435 = x57 * x74;
    const real_t x436 = u[18] * x65;
    const real_t x437 = u[18] * x67;
    const real_t x438 = u[19] * x70;
    const real_t x439 = u[19] * x67;
    const real_t x440 = u[11] * x96;
    const real_t x441 = u[12] * x98;
    const real_t x442 = u[13] * x100;
    const real_t x443 = u[14] * x100;
    const real_t x444 = u[14] * x98;
    const real_t x445 = -u[14] * x318;
    const real_t x446 = u[15] * x98;
    const real_t x447 = u[15] * x96;
    const real_t x448 = u[16] * x100;
    const real_t x449 = -u[16] * x382;
    const real_t x450 = u[16] * x96;
    const real_t x451 = -x416 * x91;
    const real_t x452 = u[17] * x98;
    const real_t x453 = u[17] * x96;
    const real_t x454 = u[18] * x100;
    const real_t x455 = u[18] * x96;
    const real_t x456 = u[19] * x100;
    const real_t x457 = u[19] * x98;
    const real_t x458 = x424 * x92;
    const real_t x459 = u[11] * x131;
    const real_t x460 = u[11] * x147;
    const real_t x461 = u[12] * x129;
    const real_t x462 = u[12] * x147;
    const real_t x463 = u[13] * x129;
    const real_t x464 = u[13] * x131;
    const real_t x465 = u[14] * x119;
    const real_t x466 = u[15] * x129;
    const real_t x467 = u[15] * x131;
    const real_t x468 = (1.0 / 15.0) * x120;
    const real_t x469 = x119 * x137;
    const real_t x470 = x119 * x136;
    const real_t x471 = u[18] * x129;
    const real_t x472 = u[18] * x147;
    const real_t x473 = u[19] * x131;
    const real_t x474 = u[19] * x147;
    const real_t x475 = u[11] * x157;
    const real_t x476 = u[12] * x159;
    const real_t x477 = u[13] * x161;
    const real_t x478 = u[14] * x161;
    const real_t x479 = u[14] * x159;
    const real_t x480 = -u[14] * x336;
    const real_t x481 = u[15] * x159;
    const real_t x482 = u[15] * x157;
    const real_t x483 = u[16] * x161;
    const real_t x484 = -u[16] * x389;
    const real_t x485 = u[16] * x157;
    const real_t x486 = -x152 * x416;
    const real_t x487 = u[17] * x159;
    const real_t x488 = u[17] * x157;
    const real_t x489 = u[18] * x161;
    const real_t x490 = u[18] * x157;
    const real_t x491 = u[19] * x161;
    const real_t x492 = u[19] * x159;
    const real_t x493 = u[11] * x193;
    const real_t x494 = u[11] * x190;
    const real_t x495 = u[12] * x188;
    const real_t x496 = u[12] * x190;
    const real_t x497 = u[13] * x188;
    const real_t x498 = u[13] * x193;
    const real_t x499 = u[15] * x188;
    const real_t x500 = u[15] * x193;
    const real_t x501 = (1.0 / 15.0) * x181;
    const real_t x502 = u[18] * x188;
    const real_t x503 = u[18] * x190;
    const real_t x504 = u[19] * x193;
    const real_t x505 = u[19] * x190;
    const real_t x506 = u[11] * x16;
    const real_t x507 = -x428;
    const real_t x508 = u[10] * x70;
    const real_t x509 = u[10] * x67;
    const real_t x510 = x508 + x509;
    const real_t x511 = u[10] * x26;
    const real_t x512 = -x430 + x511;
    const real_t x513 = u[15] * x219 + u[17] * x67;
    const real_t x514 = u[16] * x70 + u[18] * x222;
    const real_t x515 = -x462;
    const real_t x516 = u[10] * x147;
    const real_t x517 = u[10] * x131;
    const real_t x518 = x516 + x517;
    const real_t x519 = u[10] * x96;
    const real_t x520 = -x464 + x519;
    const real_t x521 = u[15] * x233 + u[17] * x147;
    const real_t x522 = u[16] * x131 + u[18] * x226;
    const real_t x523 = -x496;
    const real_t x524 = u[10] * x193;
    const real_t x525 = u[10] * x190;
    const real_t x526 = x524 + x525;
    const real_t x527 = u[10] * x157;
    const real_t x528 = -x498 + x527;
    const real_t x529 = u[15] * x246 + u[17] * x190;
    const real_t x530 = u[16] * x193 + u[18] * x249;
    const real_t x531 = u[12] * x16;
    const real_t x532 = x16 * x254;
    const real_t x533 = -x429;
    const real_t x534 = u[10] * x65;
    const real_t x535 = x509 + x534;
    const real_t x536 = u[10] * x28;
    const real_t x537 = -x426 + x536;
    const real_t x538 = u[14] * x65 + u[19] * x532;
    const real_t x539 = x16 * x262;
    const real_t x540 = -x463;
    const real_t x541 = u[10] * x129;
    const real_t x542 = x516 + x541;
    const real_t x543 = u[10] * x98;
    const real_t x544 = -x460 + x543;
    const real_t x545 = u[14] * x129 + u[19] * x539;
    const real_t x546 = x16 * x269;
    const real_t x547 = -x497;
    const real_t x548 = u[10] * x188;
    const real_t x549 = x525 + x548;
    const real_t x550 = u[10] * x159;
    const real_t x551 = -x494 + x550;
    const real_t x552 = u[14] * x188 + u[19] * x546;
    const real_t x553 = u[13] * x16;
    const real_t x554 = -x425;
    const real_t x555 = x508 + x534;
    const real_t x556 = u[10] * x30;
    const real_t x557 = -x427 + x556;
    const real_t x558 = -x459;
    const real_t x559 = x517 + x541;
    const real_t x560 = u[10] * x100;
    const real_t x561 = -x461 + x560;
    const real_t x562 = -x493;
    const real_t x563 = x524 + x548;
    const real_t x564 = u[10] * x161;
    const real_t x565 = -x495 + x564;
    const real_t x566 = u[14] * x290;
    const real_t x567 = -x220 * x566;
    const real_t x568 = u[16] * x290;
    const real_t x569 = -x220 * x568;
    const real_t x570 = u[15] * x290;
    const real_t x571 = u[16] * x35;
    const real_t x572 = u[18] * x35;
    const real_t x573 = u[18] * x290;
    const real_t x574 = -u[17] * x376 + u[19] * x376 - x21 * x566 + x21 * x570 + x220 * x506 +
                        x220 * x572 - x254 * x571 + x254 * x573 + x533 + x536;
    const real_t x575 = x18 * x35;
    const real_t x576 = -x223 * x566;
    const real_t x577 = u[17] * x290;
    const real_t x578 = -x223 * x577;
    const real_t x579 = u[15] * x35;
    const real_t x580 = -u[16] * x575 + u[19] * x575 - x18 * x566 + x18 * x573 + x223 * x506 +
                        x223 * x579 - x254 * x416 + x254 * x570 + x557 + x576 + x578;
    const real_t x581 = -x406;
    const real_t x582 = u[19] * x290;
    const real_t x583 = x254 * x582;
    const real_t x584 = u[10] * x73 + x581 + x583;
    const real_t x585 = x223 * x573;
    const real_t x586 = u[10] * x435 - 2.0 / 15.0 * u[16] * x15 * x17 * x23 + x585;
    const real_t x587 = x220 * x570;
    const real_t x588 = -x407;
    const real_t x589 = u[10] * x434 - 2.0 / 15.0 * u[17] * x15 * x20 * x23 + x587 + x588;
    const real_t x590 = -x234 * x566;
    const real_t x591 = -x234 * x568;
    const real_t x592 = x290 * x93;
    const real_t x593 = -u[14] * x592 + u[15] * x592 - u[17] * x382 + u[19] * x382 + x234 * x506 +
                        x234 * x572 - x262 * x571 + x262 * x573 + x540 + x543;
    const real_t x594 = x35 * x91;
    const real_t x595 = -x236 * x566;
    const real_t x596 = -x236 * x577;
    const real_t x597 = x236 * x35;
    const real_t x598 = -u[14] * x321 + u[15] * x597 - u[16] * x594 + u[18] * x321 + u[19] * x594 +
                        x236 * x506 - x262 * x416 + x262 * x570 + x561 + x595 + x596;
    const real_t x599 = -x441;
    const real_t x600 = x262 * x582;
    const real_t x601 = u[10] * x134 + x599 + x600;
    const real_t x602 = x236 * x573;
    const real_t x603 = u[10] * x470 - 2.0 / 15.0 * u[16] * x15 * x90 * x94 + x602;
    const real_t x604 = x234 * x570;
    const real_t x605 = -x442;
    const real_t x606 = u[10] * x469 - 2.0 / 15.0 * u[17] * x15 * x92 * x94 + x604 + x605;
    const real_t x607 = -x247 * x566;
    const real_t x608 = -x247 * x568;
    const real_t x609 = -u[17] * x389 + u[19] * x389 - x154 * x566 + x154 * x570 + x247 * x506 +
                        x247 * x572 - x269 * x571 + x269 * x573 + x547 + x550;
    const real_t x610 = x152 * x35;
    const real_t x611 = -x250 * x566;
    const real_t x612 = -x250 * x577;
    const real_t x613 = -u[16] * x610 + u[19] * x610 - x152 * x566 + x152 * x573 + x250 * x506 +
                        x250 * x579 - x269 * x416 + x269 * x570 + x565 + x611 + x612;
    const real_t x614 = -x476;
    const real_t x615 = x269 * x582;
    const real_t x616 = u[10] * x195 + x614 + x615;
    const real_t x617 = x250 * x573;
    const real_t x618 = u[10] * x345 - 2.0 / 15.0 * u[16] * x15 * x151 * x155 + x617;
    const real_t x619 = x247 * x570;
    const real_t x620 = -x477;
    const real_t x621 = u[10] * x204 - 2.0 / 15.0 * u[17] * x15 * x153 * x155 + x619 + x620;
    const real_t x622 = x223 * x572;
    const real_t x623 = -x405;
    const real_t x624 = u[10] * x80 + x581 + x587 + x623;
    const real_t x625 = -x254 * x566;
    const real_t x626 = -x223 * x568;
    const real_t x627 = -x220 * x577;
    const real_t x628 = u[19] * x35;
    const real_t x629 = x254 * x628 + x625 + x626 + x627;
    const real_t x630 = u[14] * x35;
    const real_t x631 = -u[17] * x295 + u[18] * x295 + x220 * x531 + x220 * x628 + x223 * x582 -
                        x223 * x630 - x24 * x568 + x24 * x570 + x512 + x567 + x569;
    const real_t x632 = u[18] * x597;
    const real_t x633 = -x440;
    const real_t x634 = u[10] * x140 + x599 + x604 + x633;
    const real_t x635 = -x262 * x566;
    const real_t x636 = -x236 * x568;
    const real_t x637 = -x234 * x577;
    const real_t x638 = x262 * x628 + x635 + x636 + x637;
    const real_t x639 = -u[14] * x597 - u[17] * x318 + u[18] * x318 + x234 * x531 + x234 * x628 +
                        x236 * x582 + x520 - x568 * x95 + x570 * x95 + x590 + x591;
    const real_t x640 = x250 * x572;
    const real_t x641 = -x475;
    const real_t x642 = u[10] * x202 + x614 + x619 + x641;
    const real_t x643 = -x269 * x566;
    const real_t x644 = -x250 * x568;
    const real_t x645 = -x247 * x577;
    const real_t x646 = x269 * x628 + x643 + x644 + x645;
    const real_t x647 = -u[17] * x336 + u[18] * x336 - x156 * x568 + x156 * x570 + x247 * x531 +
                        x247 * x628 + x250 * x582 - x250 * x630 + x528 + x607 + x608;
    const real_t x648 = -x254 * x568;
    const real_t x649 = -x254 * x577;
    const real_t x650 = u[10] * x85 + x585 + x623;
    const real_t x651 = -u[14] * x575 + u[18] * x575 - x18 * x568 + x18 * x582 - x223 * x416 +
                        x223 * x570 + x254 * x531 + x254 * x579 + x554 + x556;
    const real_t x652 = u[10] * x81 - 2.0 / 15.0 * u[14] * x15 * x17 * x20 + x583;
    const real_t x653 = -x262 * x568;
    const real_t x654 = -x262 * x577;
    const real_t x655 = u[10] * x468 + x602 + x633;
    const real_t x656 = -u[14] * x594 - u[16] * x321 + u[18] * x594 + u[19] * x321 - x236 * x416 +
                        x236 * x570 + x262 * x531 + x262 * x579 + x558 + x560;
    const real_t x657 = u[10] * x143 - 2.0 / 15.0 * u[14] * x15 * x90 * x92 + x600;
    const real_t x658 = -x269 * x568;
    const real_t x659 = -x269 * x577;
    const real_t x660 = u[10] * x501 + x617 + x641;
    const real_t x661 = -u[14] * x610 + u[18] * x610 - x152 * x568 + x152 * x582 - x250 * x416 +
                        x250 * x570 + x269 * x531 + x269 * x579 + x562 + x564;
    const real_t x662 = u[10] * x203 - 2.0 / 15.0 * u[14] * x15 * x151 * x153 + x615;
    const real_t x663 = u[15] * x295 - u[16] * x295 + x220 * x582 - x220 * x630 + x223 * x553 +
                        x223 * x628 + x24 * x573 - x24 * x577 + x507 + x511;
    const real_t x664 = -u[14] * x376 + u[15] * x376 - x21 * x577 + x21 * x582 - x220 * x571 +
                        x220 * x573 + x254 * x553 + x254 * x572 + x537 + x648 + x649;
    const real_t x665 = u[15] * x318 - u[16] * x318 + u[19] * x597 + x234 * x582 - x234 * x630 +
                        x236 * x553 + x515 + x519 + x573 * x95 - x577 * x95;
    const real_t x666 = -u[14] * x382 + u[15] * x382 - u[17] * x592 + u[19] * x592 - x234 * x571 +
                        x234 * x573 + x262 * x553 + x262 * x572 + x544 + x653 + x654;
    const real_t x667 = u[15] * x336 - u[16] * x336 + x156 * x573 - x156 * x577 + x247 * x582 -
                        x247 * x630 + x250 * x553 + x250 * x628 + x523 + x527;
    const real_t x668 = -u[14] * x389 + u[15] * x389 - x154 * x577 + x154 * x582 - x247 * x571 +
                        x247 * x573 + x269 * x553 + x269 * x572 + x551 + x658 + x659;
    const real_t x669 = x220 * x579 + x588;
    const real_t x670 = x234 * x579 + x605;
    const real_t x671 = x247 * x579 + x620;
    const real_t x672 = u[20] * x16;
    const real_t x673 = u[21] * x26;
    const real_t x674 = u[22] * x28;
    const real_t x675 = u[23] * x30;
    const real_t x676 = u[24] * x30;
    const real_t x677 = u[24] * x28;
    const real_t x678 = -u[24] * x295;
    const real_t x679 = u[25] * x28;
    const real_t x680 = u[25] * x26;
    const real_t x681 = u[26] * x30;
    const real_t x682 = -u[26] * x376;
    const real_t x683 = u[26] * x26;
    const real_t x684 = -u[27] * x575;
    const real_t x685 = u[27] * x28;
    const real_t x686 = u[27] * x26;
    const real_t x687 = u[28] * x30;
    const real_t x688 = u[28] * x26;
    const real_t x689 = u[29] * x30;
    const real_t x690 = u[29] * x28;
    const real_t x691 = (1.0 / 5.0) * u[20];
    const real_t x692 = u[21] * x70;
    const real_t x693 = u[21] * x67;
    const real_t x694 = u[22] * x65;
    const real_t x695 = u[22] * x67;
    const real_t x696 = u[23] * x65;
    const real_t x697 = u[23] * x70;
    const real_t x698 = u[25] * x65;
    const real_t x699 = u[25] * x70;
    const real_t x700 = u[28] * x65;
    const real_t x701 = u[28] * x67;
    const real_t x702 = u[29] * x70;
    const real_t x703 = u[29] * x67;
    const real_t x704 = u[21] * x96;
    const real_t x705 = u[22] * x98;
    const real_t x706 = u[23] * x100;
    const real_t x707 = u[24] * x100;
    const real_t x708 = u[24] * x98;
    const real_t x709 = -u[24] * x318;
    const real_t x710 = u[25] * x98;
    const real_t x711 = u[25] * x96;
    const real_t x712 = u[26] * x100;
    const real_t x713 = -u[26] * x382;
    const real_t x714 = u[26] * x96;
    const real_t x715 = -u[27] * x594;
    const real_t x716 = u[27] * x98;
    const real_t x717 = u[27] * x96;
    const real_t x718 = u[28] * x100;
    const real_t x719 = u[28] * x96;
    const real_t x720 = u[29] * x100;
    const real_t x721 = u[29] * x98;
    const real_t x722 = x691 * x92;
    const real_t x723 = u[21] * x131;
    const real_t x724 = u[21] * x147;
    const real_t x725 = u[22] * x129;
    const real_t x726 = u[22] * x147;
    const real_t x727 = u[23] * x129;
    const real_t x728 = u[23] * x131;
    const real_t x729 = u[25] * x129;
    const real_t x730 = u[25] * x131;
    const real_t x731 = u[28] * x129;
    const real_t x732 = u[28] * x147;
    const real_t x733 = u[29] * x131;
    const real_t x734 = u[29] * x147;
    const real_t x735 = u[21] * x157;
    const real_t x736 = u[22] * x159;
    const real_t x737 = u[23] * x161;
    const real_t x738 = u[24] * x161;
    const real_t x739 = u[24] * x159;
    const real_t x740 = -u[24] * x336;
    const real_t x741 = u[25] * x159;
    const real_t x742 = u[25] * x157;
    const real_t x743 = u[26] * x161;
    const real_t x744 = -u[26] * x389;
    const real_t x745 = u[26] * x157;
    const real_t x746 = -u[27] * x610;
    const real_t x747 = u[27] * x159;
    const real_t x748 = u[27] * x157;
    const real_t x749 = u[28] * x161;
    const real_t x750 = u[28] * x157;
    const real_t x751 = u[29] * x161;
    const real_t x752 = u[29] * x159;
    const real_t x753 = u[21] * x193;
    const real_t x754 = u[21] * x190;
    const real_t x755 = u[22] * x188;
    const real_t x756 = u[22] * x190;
    const real_t x757 = u[23] * x188;
    const real_t x758 = u[23] * x193;
    const real_t x759 = u[25] * x188;
    const real_t x760 = u[25] * x193;
    const real_t x761 = u[28] * x188;
    const real_t x762 = u[28] * x190;
    const real_t x763 = u[29] * x193;
    const real_t x764 = u[29] * x190;
    const real_t x765 = u[21] * x16;
    const real_t x766 = -x695;
    const real_t x767 = u[20] * x70;
    const real_t x768 = u[20] * x67;
    const real_t x769 = x767 + x768;
    const real_t x770 = u[20] * x26;
    const real_t x771 = -x697 + x770;
    const real_t x772 = u[25] * x219 + u[27] * x67;
    const real_t x773 = u[26] * x70 + u[28] * x222;
    const real_t x774 = -x726;
    const real_t x775 = u[20] * x147;
    const real_t x776 = u[20] * x131;
    const real_t x777 = x775 + x776;
    const real_t x778 = u[20] * x96;
    const real_t x779 = -x728 + x778;
    const real_t x780 = u[25] * x233 + u[27] * x147;
    const real_t x781 = u[26] * x131 + u[28] * x226;
    const real_t x782 = -x756;
    const real_t x783 = u[20] * x193;
    const real_t x784 = u[20] * x190;
    const real_t x785 = x783 + x784;
    const real_t x786 = u[20] * x157;
    const real_t x787 = -x758 + x786;
    const real_t x788 = u[25] * x246 + u[27] * x190;
    const real_t x789 = u[26] * x193 + u[28] * x249;
    const real_t x790 = u[22] * x16;
    const real_t x791 = -x696;
    const real_t x792 = u[20] * x65;
    const real_t x793 = x768 + x792;
    const real_t x794 = u[20] * x28;
    const real_t x795 = -x693 + x794;
    const real_t x796 = u[24] * x65 + u[29] * x532;
    const real_t x797 = -x727;
    const real_t x798 = u[20] * x129;
    const real_t x799 = x775 + x798;
    const real_t x800 = u[20] * x98;
    const real_t x801 = -x724 + x800;
    const real_t x802 = u[24] * x129 + u[29] * x539;
    const real_t x803 = -x757;
    const real_t x804 = u[20] * x188;
    const real_t x805 = x784 + x804;
    const real_t x806 = u[20] * x159;
    const real_t x807 = -x754 + x806;
    const real_t x808 = u[24] * x188 + u[29] * x546;
    const real_t x809 = u[23] * x16;
    const real_t x810 = -x692;
    const real_t x811 = x767 + x792;
    const real_t x812 = u[20] * x30;
    const real_t x813 = -x694 + x812;
    const real_t x814 = -x723;
    const real_t x815 = x776 + x798;
    const real_t x816 = u[20] * x100;
    const real_t x817 = -x725 + x816;
    const real_t x818 = -x753;
    const real_t x819 = x783 + x804;
    const real_t x820 = u[20] * x161;
    const real_t x821 = -x755 + x820;
    const real_t x822 = u[24] * x290;
    const real_t x823 = -x220 * x822;
    const real_t x824 = u[26] * x290;
    const real_t x825 = -x220 * x824;
    const real_t x826 = u[25] * x290;
    const real_t x827 = x254 * x35;
    const real_t x828 = x220 * x35;
    const real_t x829 = u[28] * x290;
    const real_t x830 = -u[26] * x827 - u[27] * x376 + u[28] * x828 + u[29] * x376 - x21 * x822 +
                        x21 * x826 + x220 * x765 + x254 * x829 + x791 + x794;
    const real_t x831 = -x223 * x822;
    const real_t x832 = u[27] * x290;
    const real_t x833 = -x223 * x832;
    const real_t x834 = x223 * x35;
    const real_t x835 = u[25] * x834 - u[26] * x575 - u[27] * x827 + u[29] * x575 - x18 * x822 +
                        x18 * x829 + x223 * x765 + x254 * x826 + x813 + x831 + x833;
    const real_t x836 = -x674;
    const real_t x837 = u[29] * x290;
    const real_t x838 = x254 * x837;
    const real_t x839 = u[20] * x73 + x836 + x838;
    const real_t x840 = x223 * x829;
    const real_t x841 = u[20] * x435 - 2.0 / 15.0 * u[26] * x15 * x17 * x23 + x840;
    const real_t x842 = x220 * x826;
    const real_t x843 = -x675;
    const real_t x844 = u[20] * x434 - 2.0 / 15.0 * u[27] * x15 * x20 * x23 + x842 + x843;
    const real_t x845 = -x234 * x822;
    const real_t x846 = -x234 * x824;
    const real_t x847 = x262 * x35;
    const real_t x848 = x234 * x35;
    const real_t x849 = -u[24] * x592 + u[25] * x592 - u[26] * x847 - u[27] * x382 + u[28] * x848 +
                        u[29] * x382 + x234 * x765 + x262 * x829 + x797 + x800;
    const real_t x850 = -x236 * x822;
    const real_t x851 = -x236 * x832;
    const real_t x852 = -u[24] * x321 + u[25] * x597 - u[26] * x594 - u[27] * x847 + u[28] * x321 +
                        u[29] * x594 + x236 * x765 + x262 * x826 + x817 + x850 + x851;
    const real_t x853 = -x705;
    const real_t x854 = x262 * x837;
    const real_t x855 = u[20] * x134 + x853 + x854;
    const real_t x856 = x236 * x829;
    const real_t x857 = u[20] * x470 - 2.0 / 15.0 * u[26] * x15 * x90 * x94 + x856;
    const real_t x858 = x234 * x826;
    const real_t x859 = -x706;
    const real_t x860 = u[20] * x469 - 2.0 / 15.0 * u[27] * x15 * x92 * x94 + x858 + x859;
    const real_t x861 = -x247 * x822;
    const real_t x862 = -x247 * x824;
    const real_t x863 = x269 * x35;
    const real_t x864 = x247 * x35;
    const real_t x865 = -u[26] * x863 - u[27] * x389 + u[28] * x864 + u[29] * x389 - x154 * x822 +
                        x154 * x826 + x247 * x765 + x269 * x829 + x803 + x806;
    const real_t x866 = -x250 * x822;
    const real_t x867 = -x250 * x832;
    const real_t x868 = x250 * x35;
    const real_t x869 = u[25] * x868 - u[26] * x610 - u[27] * x863 + u[29] * x610 - x152 * x822 +
                        x152 * x829 + x250 * x765 + x269 * x826 + x821 + x866 + x867;
    const real_t x870 = -x736;
    const real_t x871 = x269 * x837;
    const real_t x872 = u[20] * x195 + x870 + x871;
    const real_t x873 = x250 * x829;
    const real_t x874 = u[20] * x345 - 2.0 / 15.0 * u[26] * x15 * x151 * x155 + x873;
    const real_t x875 = x247 * x826;
    const real_t x876 = -x737;
    const real_t x877 = u[20] * x204 - 2.0 / 15.0 * u[27] * x15 * x153 * x155 + x875 + x876;
    const real_t x878 = u[28] * x834;
    const real_t x879 = -x673;
    const real_t x880 = u[20] * x80 + x836 + x842 + x879;
    const real_t x881 = -x254 * x822;
    const real_t x882 = -x223 * x824;
    const real_t x883 = -x220 * x832;
    const real_t x884 = u[29] * x827 + x881 + x882 + x883;
    const real_t x885 = -u[24] * x834 - u[27] * x295 + u[28] * x295 + u[29] * x828 + x220 * x790 +
                        x223 * x837 - x24 * x824 + x24 * x826 + x771 + x823 + x825;
    const real_t x886 = u[28] * x597;
    const real_t x887 = -x704;
    const real_t x888 = u[20] * x140 + x853 + x858 + x887;
    const real_t x889 = -x262 * x822;
    const real_t x890 = -x236 * x824;
    const real_t x891 = -x234 * x832;
    const real_t x892 = u[29] * x847 + x889 + x890 + x891;
    const real_t x893 = -u[24] * x597 - u[27] * x318 + u[28] * x318 + u[29] * x848 + x234 * x790 +
                        x236 * x837 + x779 - x824 * x95 + x826 * x95 + x845 + x846;
    const real_t x894 = u[28] * x868;
    const real_t x895 = -x735;
    const real_t x896 = u[20] * x202 + x870 + x875 + x895;
    const real_t x897 = -x269 * x822;
    const real_t x898 = -x250 * x824;
    const real_t x899 = -x247 * x832;
    const real_t x900 = u[29] * x863 + x897 + x898 + x899;
    const real_t x901 = -u[24] * x868 - u[27] * x336 + u[28] * x336 + u[29] * x864 - x156 * x824 +
                        x156 * x826 + x247 * x790 + x250 * x837 + x787 + x861 + x862;
    const real_t x902 = -x254 * x824;
    const real_t x903 = -x254 * x832;
    const real_t x904 = u[20] * x85 + x840 + x879;
    const real_t x905 = -u[24] * x575 + u[25] * x827 - u[27] * x834 + u[28] * x575 - x18 * x824 +
                        x18 * x837 + x223 * x826 + x254 * x790 + x810 + x812;
    const real_t x906 = u[20] * x81 - 2.0 / 15.0 * u[24] * x15 * x17 * x20 + x838;
    const real_t x907 = -x262 * x824;
    const real_t x908 = -x262 * x832;
    const real_t x909 = u[20] * x468 + x856 + x887;
    const real_t x910 = -u[24] * x594 + u[25] * x847 - u[26] * x321 - u[27] * x597 + u[28] * x594 +
                        u[29] * x321 + x236 * x826 + x262 * x790 + x814 + x816;
    const real_t x911 = u[20] * x143 - 2.0 / 15.0 * u[24] * x15 * x90 * x92 + x854;
    const real_t x912 = -x269 * x824;
    const real_t x913 = -x269 * x832;
    const real_t x914 = u[20] * x501 + x873 + x895;
    const real_t x915 = -u[24] * x610 + u[25] * x863 - u[27] * x868 + u[28] * x610 - x152 * x824 +
                        x152 * x837 + x250 * x826 + x269 * x790 + x818 + x820;
    const real_t x916 = u[20] * x203 - 2.0 / 15.0 * u[24] * x15 * x151 * x153 + x871;
    const real_t x917 = -u[24] * x828 + u[25] * x295 - u[26] * x295 + u[29] * x834 + x220 * x837 +
                        x223 * x809 + x24 * x829 - x24 * x832 + x766 + x770;
    const real_t x918 = -u[24] * x376 + u[25] * x376 - u[26] * x828 + u[28] * x827 - x21 * x832 +
                        x21 * x837 + x220 * x829 + x254 * x809 + x795 + x902 + x903;
    const real_t x919 = -u[24] * x848 + u[25] * x318 - u[26] * x318 + u[29] * x597 + x234 * x837 +
                        x236 * x809 + x774 + x778 + x829 * x95 - x832 * x95;
    const real_t x920 = -u[24] * x382 + u[25] * x382 - u[26] * x848 - u[27] * x592 + u[28] * x847 +
                        u[29] * x592 + x234 * x829 + x262 * x809 + x801 + x907 + x908;
    const real_t x921 = -u[24] * x864 + u[25] * x336 - u[26] * x336 + u[29] * x868 + x156 * x829 -
                        x156 * x832 + x247 * x837 + x250 * x809 + x782 + x786;
    const real_t x922 = -u[24] * x389 + u[25] * x389 - u[26] * x864 + u[28] * x863 - x154 * x832 +
                        x154 * x837 + x247 * x829 + x269 * x809 + x807 + x912 + x913;
    const real_t x923 = u[25] * x828 + x843;
    const real_t x924 = u[25] * x848 + x859;
    const real_t x925 = u[25] * x864 + x876;
    element_vector[0] +=
        -x89 * (-u[4] * x134 + u[5] * x140 - u[7] * x143 + u[9] * x134 + x101 - x102 - x103 + x104 +
                x105 + x106 - x107 + x108 - x109 + x110 - x111 - x112 + x113 + x114 + x115 + x116 +
                x117 * x118 - x117 * x141 + x118 * x119 - x119 * x141 - x120 * x142 + x120 * x146 +
                x120 * x54 + x123 + x125 + x127 + x128 + x130 + x132 - x133 * x144 - x135 * x136 -
                x135 * x137 - x136 * x144 + x138 + x139 + x145 + x148 + x149 + x150 + x22 * x91 +
                x22 * x93 + x22 * x95 + x97 + x99) -
        x89 * (-u[4] * x195 + u[5] * x202 - u[6] * x203 - u[6] * x204 - u[7] * x203 + u[9] * x195 -
               x142 * x181 + x146 * x181 + x152 * x22 + x154 * x22 + x156 * x22 + x158 + x160 +
               x162 - x163 - x164 + x165 + x166 + x167 - x168 + x169 - x170 + x171 - x172 - x173 +
               x174 + x175 + x176 + x177 + x179 * x54 + x181 * x54 + x182 * x54 + x185 + x187 +
               x189 + x191 + x192 + x194 - x196 * x197 - x196 * x205 - x197 * x198 + x199 + x200 -
               x201 * x205 + x206 + x207 + x208 + x209) -
        x89 * (-u[4] * x73 + u[5] * x80 - u[6] * x81 - u[7] * x81 + u[8] * x85 + u[9] * x73 +
               x16 * x19 + x21 * x22 + x22 * x24 + x27 + x29 + x31 - x33 - x34 + x37 + x38 + x39 -
               x41 + x43 - x44 + x46 - x48 - x49 + x50 + x51 + x52 + x53 + x54 * x56 + x54 * x58 +
               x54 * x59 + x62 + x64 + x66 + x68 + x69 + x71 - x72 * x82 - x74 * x75 - x74 * x83 -
               x75 * x76 - x76 * x82 + x77 + x78 - x79 * x83 + x84 + x86 + x87 + x88);
    element_vector[1] += -x89 * (-x151 * x239 - x153 * x239 + x156 * x210 + x165 - x167 + x170 +
                                 x173 - x175 - x208 - x209 + x240 + x243 + x245 + x248 + x251) -
                         x89 * (-x17 * x212 - x20 * x212 + x210 * x24 + x213 + x216 + x218 + x221 +
                                x224 + x37 - x39 + x44 + x49 - x51 - x87 - x88) -
                         x89 * (-u[4] * x225 * x92 - u[4] * x226 + x104 - x106 + x109 + x112 -
                                x114 - x149 - x150 + x210 * x95 + x227 + x230 + x232 + x235 + x237);
    element_vector[2] += -x89 * (-u[6] * x219 + x21 * x252 + x221 - x253 * x254 + x255 + x257 +
                                 x259 + x261 + x34 - x38 + x43 + x48 - x53 - x84 - x86) -
                         x89 * (-u[6] * x233 + x103 - x105 + x108 + x111 - x116 - x145 - x148 +
                                x235 + x252 * x93 - x253 * x262 + x263 + x265 + x267 + x268) -
                         x89 * (-u[6] * x246 + x154 * x252 + x164 - x166 + x169 + x172 - x177 -
                                x206 - x207 + x248 - x253 * x269 + x270 + x272 + x274 + x275);
    element_vector[3] += -x89 * (-u[7] * x222 + x18 * x276 + x224 - x254 * x277 + x261 + x278 +
                                 x279 + x281 + x33 + x41 + x46 - x50 - x52 - x77 - x78) -
                         x89 * (-u[7] * x226 + x102 + x107 + x110 - x113 - x115 - x138 - x139 +
                                x237 - x262 * x277 + x268 + x276 * x91 + x282 + x283 + x285) -
                         x89 * (-u[7] * x249 + x152 * x276 + x163 + x168 + x171 - x174 - x176 -
                                x199 - x200 + x251 - x269 * x277 + x275 + x286 + x287 + x289);
    element_vector[4] +=
        -x89 * (-u[0] * x295 - u[1] * x295 + (8.0 / 15.0) * u[4] * x15 * x17 * x20 +
                (4.0 / 15.0) * u[4] * x15 * x24 - x292 - x294 - x300 - x305 - x309 - x312 - x315) -
        x89 * (-u[0] * x318 - u[1] * x318 + (8.0 / 15.0) * u[4] * x15 * x90 * x92 +
               (4.0 / 15.0) * u[4] * x15 * x95 - x316 - x317 - x320 - x324 - x327 - x330 - x333) -
        x89 * (-u[0] * x336 - u[1] * x336 + (8.0 / 15.0) * u[4] * x15 * x151 * x153 +
               (4.0 / 15.0) * u[4] * x15 * x156 - x334 - x335 - x337 - x340 - x343 - x346 - x349);
    element_vector[5] += -x89 * (x279 + x300 + x350 + x352 + x356 + x357) -
                         x89 * (x283 + x320 + x358 + x360 + x364 + x365) -
                         x89 * (x287 + x337 + x366 + x368 + x372 + x373);
    element_vector[6] +=
        -x89 * (-u[0] * x376 - u[2] * x376 + (8.0 / 15.0) * u[6] * x15 * x17 * x23 +
                (4.0 / 15.0) * u[6] * x15 * x21 - x315 - x357 - x374 - x375 - x377 - x378 - x379) -
        x89 * (-u[0] * x382 - u[2] * x382 + (8.0 / 15.0) * u[6] * x15 * x90 * x94 +
               (4.0 / 15.0) * u[6] * x15 * x93 - x333 - x365 - x380 - x381 - x384 - x385 - x386) -
        x89 * (-u[0] * x389 - u[2] * x389 + (8.0 / 15.0) * u[6] * x15 * x151 * x155 +
               (4.0 / 15.0) * u[6] * x15 * x154 - x349 - x373 - x387 - x388 - x390 - x391 - x392);
    element_vector[7] +=
        -x89 * ((4.0 / 15.0) * u[7] * x15 * x152 + (8.0 / 15.0) * u[7] * x15 * x153 * x155 -
                x152 * x393 - x152 * x396 - x338 - x339 - x346 - x368 - x392 - x399 - x400) -
        x89 * ((4.0 / 15.0) * u[7] * x15 * x18 + (8.0 / 15.0) * u[7] * x15 * x20 * x23 -
               x18 * x393 - x19 * x35 - x301 - x303 - x312 - x352 - x379 - x394 - x395) -
        x89 * ((4.0 / 15.0) * u[7] * x15 * x91 + (8.0 / 15.0) * u[7] * x15 * x92 * x94 - x322 -
               x323 - x330 - x360 - x386 - x393 * x91 - x396 * x91 - x397 - x398);
    element_vector[8] += -x89 * (x257 + x305 + x356 + x377 + x394 + x401) -
                         x89 * (x265 + x324 + x364 + x384 + x397 + x402) -
                         x89 * (x272 + x340 + x372 + x390 + x399 + x403);
    element_vector[9] += -x89 * (x216 + x309 + x350 + x353 + x354 + x355 + x378 + x395 + x401) -
                         x89 * (x230 + x327 + x358 + x361 + x362 + x363 + x385 + x398 + x402) -
                         x89 * (x243 + x343 + x366 + x369 + x370 + x371 + x391 + x400 + x403);
    element_vector[10] +=
        -x89 * (-u[14] * x134 + u[15] * x140 - u[16] * x143 - u[16] * x468 - u[16] * x469 -
                u[17] * x140 - u[17] * x143 - u[17] * x470 + u[18] * x468 + u[19] * x134 +
                x117 * x458 + x119 * x458 + x120 * x424 - x136 * x465 - x137 * x465 + x404 * x91 +
                x404 * x93 + x404 * x95 + x440 + x441 + x442 - x443 - x444 + x445 + x446 + x447 -
                x448 + x449 - x450 + x451 - x452 - x453 + x454 + x455 + x456 + x457 + x459 + x460 +
                x461 + x462 + x463 + x464 + x466 + x467 + x471 + x472 + x473 + x474) -
        x89 * (-u[14] * x195 - u[14] * x204 - u[14] * x345 + u[15] * x202 - u[16] * x203 -
               u[16] * x204 - u[16] * x501 - u[17] * x202 - u[17] * x203 - u[17] * x345 +
               u[18] * x501 + u[19] * x195 + x152 * x404 + x154 * x404 + x156 * x404 + x179 * x424 +
               x181 * x424 + x182 * x424 + x475 + x476 + x477 - x478 - x479 + x480 + x481 + x482 -
               x483 + x484 - x485 + x486 - x487 - x488 + x489 + x490 + x491 + x492 + x493 + x494 +
               x495 + x496 + x497 + x498 + x499 + x500 + x502 + x503 + x504 + x505) -
        x89 * (-u[14] * x73 + u[15] * x80 - u[16] * x434 - u[16] * x81 - u[16] * x85 -
               u[17] * x435 - u[17] * x80 - u[17] * x81 + u[18] * x85 + u[19] * x73 + x18 * x404 +
               x21 * x404 + x24 * x404 + x405 + x406 + x407 - x408 - x409 + x410 + x411 + x412 -
               x413 + x414 - x415 + x417 - x418 - x419 + x420 + x421 + x422 + x423 + x424 * x56 +
               x424 * x58 + x424 * x59 + x425 + x426 + x427 + x428 + x429 + x430 - x431 * x74 -
               x431 * x76 + x432 + x433 + x436 + x437 + x438 + x439);
    element_vector[11] += -x89 * (-u[14] * x219 - u[14] * x222 + x24 * x506 + x410 - x412 + x415 +
                                  x419 - x421 - x438 - x439 + x507 + x510 + x512 + x513 + x514) -
                          x89 * (-u[14] * x226 - u[14] * x233 + x445 - x447 + x450 + x453 - x455 -
                                 x473 - x474 + x506 * x95 + x515 + x518 + x520 + x521 + x522) -
                          x89 * (-u[14] * x246 - u[14] * x249 + x156 * x506 + x480 - x482 + x485 +
                                 x488 - x490 - x504 - x505 + x523 + x526 + x528 + x529 + x530);
    element_vector[12] += -x89 * (-u[16] * x219 - u[16] * x532 + x21 * x531 + x409 - x411 + x414 +
                                  x418 - x423 - x436 - x437 + x513 + x533 + x535 + x537 + x538) -
                          x89 * (-u[16] * x233 - u[16] * x539 + x444 - x446 + x449 + x452 - x457 -
                                 x471 - x472 + x521 + x531 * x93 + x540 + x542 + x544 + x545) -
                          x89 * (-u[16] * x246 - u[16] * x546 + x154 * x531 + x479 - x481 + x484 +
                                 x487 - x492 - x502 - x503 + x529 + x547 + x549 + x551 + x552);
    element_vector[13] += -x89 * (-u[17] * x222 - u[17] * x532 + x18 * x553 + x408 + x413 + x417 -
                                  x420 - x422 - x432 - x433 + x514 + x538 + x554 + x555 + x557) -
                          x89 * (-u[17] * x226 - u[17] * x539 + x443 + x448 + x451 - x454 - x456 -
                                 x466 - x467 + x522 + x545 + x553 * x91 + x558 + x559 + x561) -
                          x89 * (-u[17] * x249 - u[17] * x546 + x152 * x553 + x478 + x483 + x486 -
                                 x489 - x491 - x499 - x500 + x530 + x552 + x562 + x563 + x565);
    element_vector[14] +=
        -x89 * (-u[10] * x295 - u[11] * x295 + (8.0 / 15.0) * u[14] * x15 * x17 * x20 +
                (4.0 / 15.0) * u[14] * x15 * x24 - x567 - x569 - x574 - x580 - x584 - x586 - x589) -
        x89 * (-u[10] * x318 - u[11] * x318 + (8.0 / 15.0) * u[14] * x15 * x90 * x92 +
               (4.0 / 15.0) * u[14] * x15 * x95 - x590 - x591 - x593 - x598 - x601 - x603 - x606) -
        x89 * (-u[10] * x336 - u[11] * x336 + (8.0 / 15.0) * u[14] * x15 * x151 * x153 +
               (4.0 / 15.0) * u[14] * x15 * x156 - x607 - x608 - x609 - x613 - x616 - x618 - x621);
    element_vector[15] += -x89 * (x555 + x574 + x622 + x624 + x629 + x631) -
                          x89 * (x559 + x593 + x632 + x634 + x638 + x639) -
                          x89 * (x563 + x609 + x640 + x642 + x646 + x647);
    element_vector[16] +=
        -x89 * (-u[10] * x376 - u[12] * x376 + (8.0 / 15.0) * u[16] * x15 * x17 * x23 +
                (4.0 / 15.0) * u[16] * x15 * x21 - x589 - x631 - x648 - x649 - x650 - x651 - x652) -
        x89 * (-u[10] * x382 - u[12] * x382 + (8.0 / 15.0) * u[16] * x15 * x90 * x94 +
               (4.0 / 15.0) * u[16] * x15 * x93 - x606 - x639 - x653 - x654 - x655 - x656 - x657) -
        x89 * (-u[10] * x389 - u[12] * x389 + (8.0 / 15.0) * u[16] * x15 * x151 * x155 +
               (4.0 / 15.0) * u[16] * x15 * x154 - x621 - x647 - x658 - x659 - x660 - x661 - x662);
    element_vector[17] += -x89 * (-u[10] * x575 - u[13] * x575 + (4.0 / 15.0) * u[17] * x15 * x18 +
                                  (8.0 / 15.0) * u[17] * x15 * x20 * x23 - x576 - x578 - x586 -
                                  x624 - x652 - x663 - x664) -
                          x89 * (-u[10] * x594 - u[13] * x594 + (4.0 / 15.0) * u[17] * x15 * x91 +
                                 (8.0 / 15.0) * u[17] * x15 * x92 * x94 - x595 - x596 - x603 -
                                 x634 - x657 - x665 - x666) -
                          x89 * (-u[10] * x610 - u[13] * x610 + (4.0 / 15.0) * u[17] * x15 * x152 +
                                 (8.0 / 15.0) * u[17] * x15 * x153 * x155 - x611 - x612 - x618 -
                                 x642 - x662 - x667 - x668);
    element_vector[18] += -x89 * (x535 + x580 + x629 + x650 + x663 + x669) -
                          x89 * (x542 + x598 + x638 + x655 + x665 + x670) -
                          x89 * (x549 + x613 + x646 + x660 + x667 + x671);
    element_vector[19] += -x89 * (x510 + x584 + x622 + x625 + x626 + x627 + x651 + x664 + x669) -
                          x89 * (x518 + x601 + x632 + x635 + x636 + x637 + x656 + x666 + x670) -
                          x89 * (x526 + x616 + x640 + x643 + x644 + x645 + x661 + x668 + x671);
    element_vector[20] +=
        -x89 * (-u[24] * x134 - u[24] * x469 - u[24] * x470 + u[25] * x140 - u[26] * x143 -
                u[26] * x468 - u[26] * x469 - u[27] * x140 - u[27] * x143 - u[27] * x470 +
                u[28] * x468 + u[29] * x134 + x117 * x722 + x119 * x722 + x120 * x691 + x672 * x91 +
                x672 * x93 + x672 * x95 + x704 + x705 + x706 - x707 - x708 + x709 + x710 + x711 -
                x712 + x713 - x714 + x715 - x716 - x717 + x718 + x719 + x720 + x721 + x723 + x724 +
                x725 + x726 + x727 + x728 + x729 + x730 + x731 + x732 + x733 + x734) -
        x89 * (-u[24] * x195 - u[24] * x204 - u[24] * x345 + u[25] * x202 - u[26] * x203 -
               u[26] * x204 - u[26] * x501 - u[27] * x202 - u[27] * x203 - u[27] * x345 +
               u[28] * x501 + u[29] * x195 + x152 * x672 + x154 * x672 + x156 * x672 + x179 * x691 +
               x181 * x691 + x182 * x691 + x735 + x736 + x737 - x738 - x739 + x740 + x741 + x742 -
               x743 + x744 - x745 + x746 - x747 - x748 + x749 + x750 + x751 + x752 + x753 + x754 +
               x755 + x756 + x757 + x758 + x759 + x760 + x761 + x762 + x763 + x764) -
        x89 * (-u[24] * x434 - u[24] * x435 - u[24] * x73 + u[25] * x80 - u[26] * x434 -
               u[26] * x81 - u[26] * x85 - u[27] * x435 - u[27] * x80 - u[27] * x81 + u[28] * x85 +
               u[29] * x73 + x18 * x672 + x21 * x672 + x24 * x672 + x56 * x691 + x58 * x691 +
               x59 * x691 + x673 + x674 + x675 - x676 - x677 + x678 + x679 + x680 - x681 + x682 -
               x683 + x684 - x685 - x686 + x687 + x688 + x689 + x690 + x692 + x693 + x694 + x695 +
               x696 + x697 + x698 + x699 + x700 + x701 + x702 + x703);
    element_vector[21] += -x89 * (-u[24] * x219 - u[24] * x222 + x24 * x765 + x678 - x680 + x683 +
                                  x686 - x688 - x702 - x703 + x766 + x769 + x771 + x772 + x773) -
                          x89 * (-u[24] * x226 - u[24] * x233 + x709 - x711 + x714 + x717 - x719 -
                                 x733 - x734 + x765 * x95 + x774 + x777 + x779 + x780 + x781) -
                          x89 * (-u[24] * x246 - u[24] * x249 + x156 * x765 + x740 - x742 + x745 +
                                 x748 - x750 - x763 - x764 + x782 + x785 + x787 + x788 + x789);
    element_vector[22] += -x89 * (-u[26] * x219 - u[26] * x532 + x21 * x790 + x677 - x679 + x682 +
                                  x685 - x690 - x700 - x701 + x772 + x791 + x793 + x795 + x796) -
                          x89 * (-u[26] * x233 - u[26] * x539 + x708 - x710 + x713 + x716 - x721 -
                                 x731 - x732 + x780 + x790 * x93 + x797 + x799 + x801 + x802) -
                          x89 * (-u[26] * x246 - u[26] * x546 + x154 * x790 + x739 - x741 + x744 +
                                 x747 - x752 - x761 - x762 + x788 + x803 + x805 + x807 + x808);
    element_vector[23] += -x89 * (-u[27] * x222 - u[27] * x532 + x18 * x809 + x676 + x681 + x684 -
                                  x687 - x689 - x698 - x699 + x773 + x796 + x810 + x811 + x813) -
                          x89 * (-u[27] * x226 - u[27] * x539 + x707 + x712 + x715 - x718 - x720 -
                                 x729 - x730 + x781 + x802 + x809 * x91 + x814 + x815 + x817) -
                          x89 * (-u[27] * x249 - u[27] * x546 + x152 * x809 + x738 + x743 + x746 -
                                 x749 - x751 - x759 - x760 + x789 + x808 + x818 + x819 + x821);
    element_vector[24] +=
        -x89 * (-u[20] * x295 - u[21] * x295 + (8.0 / 15.0) * u[24] * x15 * x17 * x20 +
                (4.0 / 15.0) * u[24] * x15 * x24 - x823 - x825 - x830 - x835 - x839 - x841 - x844) -
        x89 * (-u[20] * x318 - u[21] * x318 + (8.0 / 15.0) * u[24] * x15 * x90 * x92 +
               (4.0 / 15.0) * u[24] * x15 * x95 - x845 - x846 - x849 - x852 - x855 - x857 - x860) -
        x89 * (-u[20] * x336 - u[21] * x336 + (8.0 / 15.0) * u[24] * x15 * x151 * x153 +
               (4.0 / 15.0) * u[24] * x15 * x156 - x861 - x862 - x865 - x869 - x872 - x874 - x877);
    element_vector[25] += -x89 * (x811 + x830 + x878 + x880 + x884 + x885) -
                          x89 * (x815 + x849 + x886 + x888 + x892 + x893) -
                          x89 * (x819 + x865 + x894 + x896 + x900 + x901);
    element_vector[26] +=
        -x89 * (-u[20] * x376 - u[22] * x376 + (8.0 / 15.0) * u[26] * x15 * x17 * x23 +
                (4.0 / 15.0) * u[26] * x15 * x21 - x844 - x885 - x902 - x903 - x904 - x905 - x906) -
        x89 * (-u[20] * x382 - u[22] * x382 + (8.0 / 15.0) * u[26] * x15 * x90 * x94 +
               (4.0 / 15.0) * u[26] * x15 * x93 - x860 - x893 - x907 - x908 - x909 - x910 - x911) -
        x89 * (-u[20] * x389 - u[22] * x389 + (8.0 / 15.0) * u[26] * x15 * x151 * x155 +
               (4.0 / 15.0) * u[26] * x15 * x154 - x877 - x901 - x912 - x913 - x914 - x915 - x916);
    element_vector[27] += -x89 * (-u[20] * x575 - u[23] * x575 + (4.0 / 15.0) * u[27] * x15 * x18 +
                                  (8.0 / 15.0) * u[27] * x15 * x20 * x23 - x831 - x833 - x841 -
                                  x880 - x906 - x917 - x918) -
                          x89 * (-u[20] * x594 - u[23] * x594 + (4.0 / 15.0) * u[27] * x15 * x91 +
                                 (8.0 / 15.0) * u[27] * x15 * x92 * x94 - x850 - x851 - x857 -
                                 x888 - x911 - x919 - x920) -
                          x89 * (-u[20] * x610 - u[23] * x610 + (4.0 / 15.0) * u[27] * x15 * x152 +
                                 (8.0 / 15.0) * u[27] * x15 * x153 * x155 - x866 - x867 - x874 -
                                 x896 - x916 - x921 - x922);
    element_vector[28] += -x89 * (x793 + x835 + x884 + x904 + x917 + x923) -
                          x89 * (x799 + x852 + x892 + x909 + x919 + x924) -
                          x89 * (x805 + x869 + x900 + x914 + x921 + x925);
    element_vector[29] += -x89 * (x769 + x839 + x878 + x881 + x882 + x883 + x905 + x918 + x923) -
                          x89 * (x777 + x855 + x886 + x889 + x890 + x891 + x910 + x920 + x924) -
                          x89 * (x785 + x872 + x894 + x897 + x898 + x899 + x915 + x922 + x925);
}



// static SFEM_INLINE void tet10_explict_momentum_rhs_kernel(const real_t px0,
//                                                          const real_t px1,
//                                                          const real_t px2,
//                                                          const real_t py0,
//                                                          const real_t py1,
//                                                          const real_t py2,
//                                                          const real_t dt,
//                                                          const real_t nu,
//                                                          const real_t convonoff,
//                                                          real_t *const SFEM_RESTRICT u,
//                                                          real_t *const SFEM_RESTRICT
//                                                              element_vector) {}

// static SFEM_INLINE void tet10_add_momentum_rhs_kernel(const real_t px0,
//                                                      const real_t px1,
//                                                      const real_t px2,
//                                                      const real_t py0,
//                                                      const real_t py1,
//                                                      const real_t py2,
//                                                      real_t *const SFEM_RESTRICT u,
//                                                      real_t *const SFEM_RESTRICT element_vector)
//                                                      {}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols10(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 10; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(10)
        for (int d = 0; d < 10; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(10)
            for (int d = 0; d < 10; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void tet4_tet10_divergence(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const elems,
                           geom_t **const points,
                           const real_t dt,
                           const real_t rho,
                           const real_t nu,
                           real_t **const SFEM_RESTRICT vel,
                           real_t *const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int element_nnodes = 10;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            real_t element_vector[4];
            real_t element_vel[10 * 3];

#pragma unroll(10)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_vel[b * element_nnodes + enode] = vel[b][dof];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet4_tet10_divergence_rhs_kernel(
                // X-coordinates
                x0,
                x1,
                x2,
                x3,
                // Y-coordinates
                y0,
                y1,
                y2,
                y3,
                // Z-coordinates
                z0,
                z1,
                z2,
                z3,
                dt,
                rho,
                //  buffers
                element_vel,
                element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];
#pragma omp atomic update
                f[dof_i] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tet10_naviers_stokes.c: tet10_explict_momentum_tentative\t%g seconds\n", tock -
    // tick);
}

void tet10_tet4_correction(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const elems,
                           geom_t **const points,
                           const real_t dt,
                           const real_t rho,
                           real_t *const SFEM_RESTRICT p,
                           real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int element_nnodes = 10;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            real_t element_vector[10 * 3];
            real_t element_pressure[4];

#pragma unroll(10)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(3)
            for (int enode = 0; enode < 4; ++enode) {
                idx_t dof = ev[enode];
                element_pressure[enode] = p[dof];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet10_tet4_rhs_correction_kernel(
                // X-coordinates
                x0,
                x1,
                x2,
                x3,
                // Y-coordinates
                y0,
                y1,
                y2,
                y3,
                // Z-coordinates
                z0,
                z1,
                z2,
                z3,
                dt,
                rho,
                //  buffers
                element_pressure,
                element_vector);

            for (int b = 0; b < n_vars; ++b) {
#pragma unroll(10)
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
                    const int iii = b * element_nnodes + edof_i;
                    assert(element_vector[iii] == element_vector[iii]);
#pragma omp atomic update
                    values[b][dof_i] += element_vector[iii];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tet10_naviers_stokes.c: tet10_explict_momentum_tentative\t%g seconds\n", tock -
    // tick);
}

void tet10_momentum_lhs_scalar_crs(const ptrdiff_t nelements,
                                   const ptrdiff_t nnodes,
                                   idx_t **const elems,
                                   geom_t **const points,
                                   const real_t dt,
                                   const real_t nu,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT colidx,
                                   real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            idx_t ks[10];
            real_t element_matrix[10 * 10];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            tet10_momentum_lhs_scalar_kernel(
                // X-coordinates
                x0,
                x1,
                x2,
                x3,
                // Y-coordinates
                y0,
                y1,
                y2,
                y3,
                // Z-coordinates
                z0,
                z1,
                z2,
                z3,
                dt,
                nu,
                element_matrix);

            for (int edof_i = 0; edof_i < 10; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols10(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 10];

#pragma unroll(10)
                for (int edof_j = 0; edof_j < 10; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_laplacian.c: tet4_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}

void tet10_explict_momentum_tentative(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const elems,
                                      geom_t **const points,
                                      const real_t dt,
                                      const real_t nu,
                                      const real_t convonoff,
                                      real_t **const SFEM_RESTRICT vel,
                                      real_t **const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int element_nnodes = 10;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            real_t element_vector[10 * 3];
            real_t element_vel[10 * 3];

#pragma unroll(10)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    assert(vel[b][dof] == vel[b][dof]);
                    element_vel[b * element_nnodes + enode] = vel[b][dof];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];
            const real_t x3 = points[0][i3];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];
            const real_t y3 = points[1][i3];

            const real_t z0 = points[2][i0];
            const real_t z1 = points[2][i1];
            const real_t z2 = points[2][i2];
            const real_t z3 = points[2][i3];

            memset(element_vector, 0, 10 * 3 * sizeof(real_t));

            tet10_add_diffusion_rhs_kernel(x0,
                                           x1,
                                           x2,
                                           x3,
                                           // Y coords
                                           y0,
                                           y1,
                                           y2,
                                           y3,
                                           // Z coords
                                           z0,
                                           z1,
                                           z2,
                                           z3,
                                           dt,
                                           nu,
                                           //  buffers
                                           element_vel,
                                           element_vector);

            if (convonoff != 0) {
                tet10_add_convection_rhs_kernel(x0,
                                                x1,
                                                x2,
                                                x3,
                                                // Y coords
                                                y0,
                                                y1,
                                                y2,
                                                y3,
                                                // Z coords
                                                z0,
                                                z1,
                                                z2,
                                                z3,
                                                dt,
                                                nu,
                                                //  buffers
                                                element_vel,
                                                element_vector);
            }

            for (int d1 = 0; d1 < n_vars; d1++) {
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
                    int iii = d1 * element_nnodes + edof_i;
                    assert(element_vector[iii] == element_vector[iii]);
#pragma omp atomic update
                    f[d1][dof_i] += element_vector[iii];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tet10_naviers_stokes.c: tet10_explict_momentum_tentative\t%g seconds\n", tock -
    // tick);
}
