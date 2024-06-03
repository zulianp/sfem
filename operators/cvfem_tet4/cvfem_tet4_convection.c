#include "cvfem_tet4_convection.h"

#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE real_t det_jacobian(  // X
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // Y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // Z
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = -px0 + px2;
    const real_t x4 = -py0 + py3;
    const real_t x5 = -pz0 + pz1;
    const real_t x6 = -px0 + px3;
    const real_t x7 = -py0 + py1;
    const real_t x8 = -pz0 + pz2;
    return x0 * x1 * x2 - x0 * x4 * x8 - x1 * x5 * x6 - x2 * x3 * x7 + x3 * x4 * x5 + x6 * x7 * x8;
}

static SFEM_INLINE void cvfem_tet4_convection_assemble_hessian_kernel(
    // X
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // Y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // Z
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3,
    // Velocity
    const real_t *const SFEM_RESTRICT vx,
    const real_t *const SFEM_RESTRICT vy,
    const real_t *const SFEM_RESTRICT vz,
    // Output
    real_t *const SFEM_RESTRICT element_matrix) {
    real_t J[9];
    {
        J[0] = -px0 + px1;
        J[1] = -px0 + px2;
        J[2] = -px0 + px3;
        J[3] = -py0 + py1;
        J[4] = -py0 + py2;
        J[5] = -py0 + py3;
        J[6] = -pz0 + pz1;
        J[7] = -pz0 + pz2;
        J[8] = -pz0 + pz3;
    }

    const real_t x0 = 17 * vx[0];
    const real_t x1 = 17 * vx[1];
    const real_t x2 = 7 * vx[2];
    const real_t x3 = 7 * vx[3];
    const real_t x4 = 2 * J[4];
    const real_t x5 = J[3] - x4;
    const real_t x6 = 2 * J[6];
    const real_t x7 = x6 - 1;
    const real_t x8 = 2 * J[3];
    const real_t x9 = x8 - 1;
    const real_t x10 = 2 * J[8];
    const real_t x11 = -x10;
    const real_t x12 = J[6] + x11;
    const real_t x13 = 2 * J[5];
    const real_t x14 = -x13;
    const real_t x15 = J[3] + x14;
    const real_t x16 = 2 * J[7];
    const real_t x17 = J[6] - x16;
    const real_t x18 = 17 * vy[0];
    const real_t x19 = 17 * vy[1];
    const real_t x20 = 7 * vy[2];
    const real_t x21 = 7 * vy[3];
    const real_t x22 = 2 * J[1];
    const real_t x23 = J[0] - x22;
    const real_t x24 = 2 * J[0];
    const real_t x25 = x24 - 1;
    const real_t x26 = 2 * J[2];
    const real_t x27 = -x26;
    const real_t x28 = J[0] + x27;
    const real_t x29 = 17 * vz[0];
    const real_t x30 = 17 * vz[1];
    const real_t x31 = 7 * vz[2];
    const real_t x32 = 7 * vz[3];
    const real_t x33 =
        (1.0 / 2304.0) * (x0 + x1 + x2 + x3) * (x12 * x9 - x15 * x7 - x17 * x9 + x5 * x7) -
        1.0 / 2304.0 * (x18 + x19 + x20 + x21) * (x12 * x25 - x17 * x25 + x23 * x7 - x28 * x7) +
        (1.0 / 2304.0) * (x29 + x30 + x31 + x32) * (x15 * x25 + x23 * x9 - x25 * x5 - x28 * x9);
    const real_t x34 = ((x33 < 0) ? (0) : (x33));
    const real_t x35 = 7 * vx[1];
    const real_t x36 = x0 + x35;
    const real_t x37 = 17 * vx[2];
    const real_t x38 = x3 + x37;
    const real_t x39 = J[4] + x14;
    const real_t x40 = x16 - 1;
    const real_t x41 = x4 - 1;
    const real_t x42 = J[7] + x11;
    const real_t x43 = -J[4] + x8;
    const real_t x44 = -J[7] + x6;
    const real_t x45 = 7 * vy[1];
    const real_t x46 = x18 + x45;
    const real_t x47 = 17 * vy[2];
    const real_t x48 = x21 + x47;
    const real_t x49 = J[1] + x27;
    const real_t x50 = x22 - 1;
    const real_t x51 = -J[1] + x24;
    const real_t x52 = 7 * vz[1];
    const real_t x53 = x29 + x52;
    const real_t x54 = 17 * vz[2];
    const real_t x55 = x32 + x54;
    const real_t x56 =
        (1.0 / 2304.0) * (x36 + x38) * (x39 * x40 + x40 * x43 - x41 * x42 - x41 * x44) -
        1.0 / 2304.0 * (x46 + x48) * (x40 * x49 + x40 * x51 - x42 * x50 - x44 * x50) +
        (1.0 / 2304.0) * (x53 + x55) * (-x39 * x50 + x41 * x49 + x41 * x51 - x43 * x50);
    const real_t x57 = ((x56 < 0) ? (0) : (x56));
    const real_t x58 = 17 * vx[3];
    const real_t x59 = x2 + x58;
    const real_t x60 = x10 - 1;
    const real_t x61 = -J[5];
    const real_t x62 = x61 + x8;
    const real_t x63 = x13 - 1;
    const real_t x64 = -J[8];
    const real_t x65 = x16 + x64;
    const real_t x66 = x4 + x61;
    const real_t x67 = x6 + x64;
    const real_t x68 = 17 * vy[3];
    const real_t x69 = x20 + x68;
    const real_t x70 = -J[2];
    const real_t x71 = x24 + x70;
    const real_t x72 = x26 - 1;
    const real_t x73 = x22 + x70;
    const real_t x74 = 17 * vz[3];
    const real_t x75 = x31 + x74;
    const real_t x76 =
        (1.0 / 2304.0) * (x36 + x59) * (x60 * x62 - x60 * x66 + x63 * x65 - x63 * x67) -
        1.0 / 2304.0 * (x46 + x69) * (x60 * x71 - x60 * x73 + x65 * x72 - x67 * x72) +
        (1.0 / 2304.0) * (x53 + x75) * (-x62 * x72 + x63 * x71 - x63 * x73 + x66 * x72);
    const real_t x77 = ((x76 > 0) ? (0) : (-x76));
    const real_t x78 = ((x33 > 0) ? (0) : (-x33));
    const real_t x79 = ((x56 > 0) ? (0) : (-x56));
    const real_t x80 = ((x76 < 0) ? (0) : (x76));
    const real_t x81 = 7 * vx[0];
    const real_t x82 = x1 + x81;
    const real_t x83 = J[3] + J[5];
    const real_t x84 = x6 + x60;
    const real_t x85 = J[6] + J[8];
    const real_t x86 = x63 + x8;
    const real_t x87 = J[8] + x17;
    const real_t x88 = J[5] + x5;
    const real_t x89 = 7 * vy[0];
    const real_t x90 = x19 + x89;
    const real_t x91 = J[0] + J[2];
    const real_t x92 = x24 + x72;
    const real_t x93 = J[2] + x23;
    const real_t x94 = 7 * vz[0];
    const real_t x95 = x30 + x94;
    const real_t x96 =
        (1.0 / 2304.0) * (x59 + x82) * (x83 * x84 - x84 * x88 - x85 * x86 + x86 * x87) -
        1.0 / 2304.0 * (x69 + x90) * (x84 * x91 - x84 * x93 - x85 * x92 + x87 * x92) +
        (1.0 / 2304.0) * (x75 + x95) * (-x83 * x92 + x86 * x91 - x86 * x93 + x88 * x92);
    const real_t x97 = ((x96 > 0) ? (0) : (-x96));
    const real_t x98 = J[3] + J[4];
    const real_t x99 = x40 + x6;
    const real_t x100 = J[6] + J[7];
    const real_t x101 = x41 + x8;
    const real_t x102 = J[6] + x42;
    const real_t x103 = J[3] + x39;
    const real_t x104 = J[0] + J[1];
    const real_t x105 = x24 + x50;
    const real_t x106 = J[0] + x49;
    const real_t x107 =
        (1.0 / 2304.0) * (x38 + x82) * (-x100 * x101 + x101 * x102 - x103 * x99 + x98 * x99) -
        1.0 / 2304.0 * (x48 + x90) * (-x100 * x105 + x102 * x105 + x104 * x99 - x106 * x99) +
        (1.0 / 2304.0) * (x55 + x95) * (x101 * x104 - x101 * x106 + x103 * x105 - x105 * x98);
    const real_t x108 = ((x107 < 0) ? (0) : (x107));
    const real_t x109 = ((x107 > 0) ? (0) : (-x107));
    const real_t x110 = ((x96 < 0) ? (0) : (x96));
    const real_t x111 = J[4] + J[5];
    const real_t x112 = x16 + x60;
    const real_t x113 = J[7] + J[8];
    const real_t x114 = x4 + x63;
    const real_t x115 = x113 - x6;
    const real_t x116 = x111 - x8;
    const real_t x117 = J[1] + J[2];
    const real_t x118 = x22 + x72;
    const real_t x119 = x117 - x24;
    const real_t x120 = (1.0 / 2304.0) * (x35 + x37 + x58 + x81) *
                            (x111 * x112 - x112 * x116 - x113 * x114 + x114 * x115) -
                        1.0 / 2304.0 * (x45 + x47 + x68 + x89) *
                            (x112 * x117 - x112 * x119 - x113 * x118 + x115 * x118) +
                        (1.0 / 2304.0) * (x52 + x54 + x74 + x94) *
                            (-x111 * x118 + x114 * x117 - x114 * x119 + x116 * x118);
    const real_t x121 = ((x120 < 0) ? (0) : (x120));
    const real_t x122 = ((x120 > 0) ? (0) : (-x120));
    element_matrix[0] = -x34 - x57 - x77;
    element_matrix[1] = x78;
    element_matrix[2] = x79;
    element_matrix[3] = x80;
    element_matrix[4] = x34;
    element_matrix[5] = -x108 - x78 - x97;
    element_matrix[6] = x109;
    element_matrix[7] = x110;
    element_matrix[8] = x57;
    element_matrix[9] = x108;
    element_matrix[10] = -x109 - x121 - x79;
    element_matrix[11] = x122;
    element_matrix[12] = x77;
    element_matrix[13] = x97;
    element_matrix[14] = x121;
    element_matrix[15] = -x110 - x122 - x80;
}

static SFEM_INLINE void cvfem_tet4_convection_apply_kernel(
    // X
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // Y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // Z
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3,
    // Velocity
    const real_t *const SFEM_RESTRICT vx,
    const real_t *const SFEM_RESTRICT vy,
    const real_t *const SFEM_RESTRICT vz,
    // Input
    const real_t *const SFEM_RESTRICT x,
    // Output
    real_t *const SFEM_RESTRICT element_vector) {
    real_t J[9];
    {
        J[0] = -px0 + px1;
        J[1] = -px0 + px2;
        J[2] = -px0 + px3;
        J[3] = -py0 + py1;
        J[4] = -py0 + py2;
        J[5] = -py0 + py3;
        J[6] = -pz0 + pz1;
        J[7] = -pz0 + pz2;
        J[8] = -pz0 + pz3;
    }

    const real_t x0 = 17 * vx[0];
    const real_t x1 = 17 * vx[1];
    const real_t x2 = 7 * vx[2];
    const real_t x3 = 7 * vx[3];
    const real_t x4 = 2 * J[4];
    const real_t x5 = J[3] - x4;
    const real_t x6 = 2 * J[6];
    const real_t x7 = x6 - 1;
    const real_t x8 = 2 * J[3];
    const real_t x9 = x8 - 1;
    const real_t x10 = 2 * J[8];
    const real_t x11 = -x10;
    const real_t x12 = J[6] + x11;
    const real_t x13 = 2 * J[5];
    const real_t x14 = -x13;
    const real_t x15 = J[3] + x14;
    const real_t x16 = 2 * J[7];
    const real_t x17 = J[6] - x16;
    const real_t x18 = 17 * vy[0];
    const real_t x19 = 17 * vy[1];
    const real_t x20 = 7 * vy[2];
    const real_t x21 = 7 * vy[3];
    const real_t x22 = 2 * J[1];
    const real_t x23 = J[0] - x22;
    const real_t x24 = 2 * J[0];
    const real_t x25 = x24 - 1;
    const real_t x26 = 2 * J[2];
    const real_t x27 = -x26;
    const real_t x28 = J[0] + x27;
    const real_t x29 = 17 * vz[0];
    const real_t x30 = 17 * vz[1];
    const real_t x31 = 7 * vz[2];
    const real_t x32 = 7 * vz[3];
    const real_t x33 =
        (1.0 / 2304.0) * (x0 + x1 + x2 + x3) * (x12 * x9 - x15 * x7 - x17 * x9 + x5 * x7) -
        1.0 / 2304.0 * (x18 + x19 + x20 + x21) * (x12 * x25 - x17 * x25 + x23 * x7 - x28 * x7) +
        (1.0 / 2304.0) * (x29 + x30 + x31 + x32) * (x15 * x25 + x23 * x9 - x25 * x5 - x28 * x9);
    const real_t x34 = ((x33 > 0) ? (0) : (-x33));
    const real_t x35 = 7 * vx[1];
    const real_t x36 = x0 + x35;
    const real_t x37 = 17 * vx[2];
    const real_t x38 = x3 + x37;
    const real_t x39 = J[4] + x14;
    const real_t x40 = x16 - 1;
    const real_t x41 = x4 - 1;
    const real_t x42 = J[7] + x11;
    const real_t x43 = -J[4] + x8;
    const real_t x44 = -J[7] + x6;
    const real_t x45 = 7 * vy[1];
    const real_t x46 = x18 + x45;
    const real_t x47 = 17 * vy[2];
    const real_t x48 = x21 + x47;
    const real_t x49 = J[1] + x27;
    const real_t x50 = x22 - 1;
    const real_t x51 = -J[1] + x24;
    const real_t x52 = 7 * vz[1];
    const real_t x53 = x29 + x52;
    const real_t x54 = 17 * vz[2];
    const real_t x55 = x32 + x54;
    const real_t x56 =
        (1.0 / 2304.0) * (x36 + x38) * (x39 * x40 + x40 * x43 - x41 * x42 - x41 * x44) -
        1.0 / 2304.0 * (x46 + x48) * (x40 * x49 + x40 * x51 - x42 * x50 - x44 * x50) +
        (1.0 / 2304.0) * (x53 + x55) * (-x39 * x50 + x41 * x49 + x41 * x51 - x43 * x50);
    const real_t x57 = ((x56 > 0) ? (0) : (-x56));
    const real_t x58 = 17 * vx[3];
    const real_t x59 = x2 + x58;
    const real_t x60 = x10 - 1;
    const real_t x61 = -J[5];
    const real_t x62 = x61 + x8;
    const real_t x63 = x13 - 1;
    const real_t x64 = -J[8];
    const real_t x65 = x16 + x64;
    const real_t x66 = x4 + x61;
    const real_t x67 = x6 + x64;
    const real_t x68 = 17 * vy[3];
    const real_t x69 = x20 + x68;
    const real_t x70 = -J[2];
    const real_t x71 = x24 + x70;
    const real_t x72 = x26 - 1;
    const real_t x73 = x22 + x70;
    const real_t x74 = 17 * vz[3];
    const real_t x75 = x31 + x74;
    const real_t x76 =
        (1.0 / 2304.0) * (x36 + x59) * (x60 * x62 - x60 * x66 + x63 * x65 - x63 * x67) -
        1.0 / 2304.0 * (x46 + x69) * (x60 * x71 - x60 * x73 + x65 * x72 - x67 * x72) +
        (1.0 / 2304.0) * (x53 + x75) * (-x62 * x72 + x63 * x71 - x63 * x73 + x66 * x72);
    const real_t x77 = ((x76 < 0) ? (0) : (x76));
    const real_t x78 = ((x33 < 0) ? (0) : (x33));
    const real_t x79 = ((x56 < 0) ? (0) : (x56));
    const real_t x80 = ((x76 > 0) ? (0) : (-x76));
    const real_t x81 = 7 * vx[0];
    const real_t x82 = x1 + x81;
    const real_t x83 = J[3] + J[4];
    const real_t x84 = x40 + x6;
    const real_t x85 = J[6] + J[7];
    const real_t x86 = x41 + x8;
    const real_t x87 = J[6] + x42;
    const real_t x88 = J[3] + x39;
    const real_t x89 = 7 * vy[0];
    const real_t x90 = x19 + x89;
    const real_t x91 = J[0] + J[1];
    const real_t x92 = x24 + x50;
    const real_t x93 = J[0] + x49;
    const real_t x94 = 7 * vz[0];
    const real_t x95 = x30 + x94;
    const real_t x96 =
        (1.0 / 2304.0) * (x38 + x82) * (x83 * x84 - x84 * x88 - x85 * x86 + x86 * x87) -
        1.0 / 2304.0 * (x48 + x90) * (x84 * x91 - x84 * x93 - x85 * x92 + x87 * x92) +
        (1.0 / 2304.0) * (x55 + x95) * (-x83 * x92 + x86 * x91 - x86 * x93 + x88 * x92);
    const real_t x97 = ((x96 > 0) ? (0) : (-x96));
    const real_t x98 = J[3] + J[5];
    const real_t x99 = x6 + x60;
    const real_t x100 = J[6] + J[8];
    const real_t x101 = x63 + x8;
    const real_t x102 = J[8] + x17;
    const real_t x103 = J[5] + x5;
    const real_t x104 = J[0] + J[2];
    const real_t x105 = x24 + x72;
    const real_t x106 = J[2] + x23;
    const real_t x107 =
        (1.0 / 2304.0) * (x59 + x82) * (-x100 * x101 + x101 * x102 - x103 * x99 + x98 * x99) -
        1.0 / 2304.0 * (x69 + x90) * (-x100 * x105 + x102 * x105 + x104 * x99 - x106 * x99) +
        (1.0 / 2304.0) * (x75 + x95) * (x101 * x104 - x101 * x106 + x103 * x105 - x105 * x98);
    const real_t x108 = ((x107 < 0) ? (0) : (x107));
    const real_t x109 = ((x107 > 0) ? (0) : (-x107));
    const real_t x110 = ((x96 < 0) ? (0) : (x96));
    const real_t x111 = J[4] + J[5];
    const real_t x112 = x16 + x60;
    const real_t x113 = J[7] + J[8];
    const real_t x114 = x4 + x63;
    const real_t x115 = x113 - x6;
    const real_t x116 = x111 - x8;
    const real_t x117 = J[1] + J[2];
    const real_t x118 = x22 + x72;
    const real_t x119 = x117 - x24;
    const real_t x120 = (1.0 / 2304.0) * (x35 + x37 + x58 + x81) *
                            (x111 * x112 - x112 * x116 - x113 * x114 + x114 * x115) -
                        1.0 / 2304.0 * (x45 + x47 + x68 + x89) *
                            (x112 * x117 - x112 * x119 - x113 * x118 + x115 * x118) +
                        (1.0 / 2304.0) * (x52 + x54 + x74 + x94) *
                            (-x111 * x118 + x114 * x117 - x114 * x119 + x116 * x118);
    const real_t x121 = ((x120 > 0) ? (0) : (-x120));
    const real_t x122 = ((x120 < 0) ? (0) : (x120));
    element_vector[0] = x34 * x[1] + x57 * x[2] + x77 * x[3] + x[0] * (-x78 - x79 - x80);
    element_vector[1] = x108 * x[3] + x78 * x[0] + x97 * x[2] + x[1] * (-x109 - x110 - x34);
    element_vector[2] = x110 * x[1] + x121 * x[3] + x79 * x[0] + x[2] * (-x122 - x57 - x97);
    element_vector[3] = x109 * x[1] + x122 * x[2] + x80 * x[0] + x[3] * (-x108 - x121 - x77);
}

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

static SFEM_INLINE void find_cols4(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(4)
            for (int d = 0; d < 4; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void cvfem_tet4_convection_assemble_hessian(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            real_t **const SFEM_RESTRICT velocity,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];
            real_t element_matrix[4 * 4];
            real_t vx[4];
            real_t vy[4];
            real_t vz[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vx[v] = velocity[0][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vy[v] = velocity[1][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vz[v] = velocity[2][ev[v]];
            }

            cvfem_tet4_convection_assemble_hessian_kernel(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                vx,
                vy,
                vz,
                element_matrix);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols4(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 4];

#pragma unroll(4)
                for (int edof_j = 0; edof_j < 4; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }
}

void cvfem_tet4_convection_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 real_t **const SFEM_RESTRICT velocity,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];

            real_t vx[4];
            real_t vy[4];
            real_t vz[4];
            real_t element_u[4];
            real_t element_vector[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vx[v] = velocity[0][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vy[v] = velocity[1][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vz[v] = velocity[2][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                element_u[v] = u[ev[v]];
            }

            cvfem_tet4_convection_apply_kernel(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                // Velocity
                vx,
                vy,
                vz,
                // Input
                element_u,
                // Output
                element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }
}

void cvfem_tet4_cv_volumes(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            const real_t measure = det_jacobian(
                                       // X-coordinates
                                       x[ev[0]],
                                       x[ev[1]],
                                       x[ev[2]],
                                       x[ev[3]],
                                       // Y-coordinates
                                       y[ev[0]],
                                       y[ev[1]],
                                       y[ev[2]],
                                       y[ev[3]],
                                       // Z-coordinates
                                       z[ev[0]],
                                       z[ev[1]],
                                       z[ev[2]],
                                       z[ev[3]]) /
                                   4;

            assert(measure > 0);
            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += measure;
            }
        }
    }
}
