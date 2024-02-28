// #include "tet4_viscous_power_density_curnier.h"

// #include <assert.h>
// #include <math.h>
// #include <stdio.h>

// #include <mpi.h>

// #include "crs_graph.h"
// #include "sfem_vec.h"
// #include "sortreduce.h"

// static SFEM_INLINE void viscous_power_density_curnier_value(const real_t mu,
//                                                             const real_t lambda,
//                                                             const real_t eta,
//                                                             const real_t px0,
//                                                             const real_t px1,
//                                                             const real_t px2,
//                                                             const real_t px3,
//                                                             const real_t py0,
//                                                             const real_t py1,
//                                                             const real_t py2,
//                                                             const real_t py3,
//                                                             const real_t pz0,
//                                                             const real_t pz1,
//                                                             const real_t pz2,
//                                                             const real_t pz3,
//                                                             const real_t *const SFEM_RESTRICT u_old,
//                                                             const real_t *const SFEM_RESTRICT u,
//                                                             real_t *const SFEM_RESTRICT
//                                                                 element_energy) {
//     const real_t x0 = pz0 - pz3;
//     const real_t x1 = -x0;
//     const real_t x2 = py0 - py2;
//     const real_t x3 = -x2;
//     const real_t x4 = px0 - px1;
//     const real_t x5 = -1.0 / 6.0 * x4;
//     const real_t x6 = py0 - py3;
//     const real_t x7 = -x6;
//     const real_t x8 = pz0 - pz2;
//     const real_t x9 = -x8;
//     const real_t x10 = py0 - py1;
//     const real_t x11 = -x10;
//     const real_t x12 = px0 - px2;
//     const real_t x13 = -1.0 / 6.0 * x12;
//     const real_t x14 = pz0 - pz1;
//     const real_t x15 = -x14;
//     const real_t x16 = px0 - px3;
//     const real_t x17 = -1.0 / 6.0 * x16;
//     const real_t x18 = x12 * x6;
//     const real_t x19 = x16 * x2;
//     const real_t x20 = x18 - x19;
//     const real_t x21 = -x20;
//     const real_t x22 = x2 * x4;
//     const real_t x23 = x10 * x16;
//     const real_t x24 = x4 * x6;
//     const real_t x25 = x10 * x12;
//     const real_t x26 = 1.0 / (x0 * x22 - x0 * x25 + x14 * x18 - x14 * x19 + x23 * x8 - x24 * x8);
//     const real_t x27 = u[3] * x26;
//     const real_t x28 = -x23 + x24;
//     const real_t x29 = u[6] * x26;
//     const real_t x30 = x22 - x25;
//     const real_t x31 = -x30;
//     const real_t x32 = u[9] * x26;
//     const real_t x33 = x20 + x23 - x24 + x30;
//     const real_t x34 = u[0] * x26;
//     const real_t x35 = x21 * x27 + x28 * x29 + x31 * x32 + x33 * x34;
//     const real_t x36 = x0 * x12 - x16 * x8;
//     const real_t x37 = x0 * x4;
//     const real_t x38 = x14 * x16;
//     const real_t x39 = -x37 + x38;
//     const real_t x40 = -x12 * x14 + x4 * x8;
//     const real_t x41 = -x36 + x37 - x38 - x40;
//     const real_t x42 = x27 * x36 + x29 * x39 + x32 * x40 + x34 * x41;
//     const real_t x43 = u[10] * x26;
//     const real_t x44 = u[4] * x26;
//     const real_t x45 = u[7] * x26;
//     const real_t x46 = u[1] * x26;
//     const real_t x47 = x21 * x44 + x28 * x45 + x31 * x43 + x33 * x46;
//     const real_t x48 = x10 * x8 - x14 * x2;
//     const real_t x49 = -x48;
//     const real_t x50 = x0 * x2 - x6 * x8;
//     const real_t x51 = -x50;
//     const real_t x52 = x0 * x10;
//     const real_t x53 = x14 * x6;
//     const real_t x54 = x52 - x53;
//     const real_t x55 = x48 + x50 - x52 + x53;
//     const real_t x56 = x43 * x49 + x44 * x51 + x45 * x54 + x46 * x55;
//     const real_t x57 = x26 * x40;
//     const real_t x58 = x26 * x36;
//     const real_t x59 = x26 * x39;
//     const real_t x60 = x26 * x41;
//     const real_t x61 = u[11] * x57 + u[2] * x60 + u[5] * x58 + u[8] * x59;
//     const real_t x62 = x26 * x49;
//     const real_t x63 = x26 * x51;
//     const real_t x64 = x26 * x54;
//     const real_t x65 = x26 * x55;
//     const real_t x66 = u[11] * x62 + u[2] * x65 + u[5] * x63 + u[8] * x64;
//     const real_t x67 = x27 * x51 + x29 * x54 + x32 * x49 + x34 * x55 + 1;
//     const real_t x68 = x36 * x44 + x39 * x45 + x40 * x43 + x41 * x46 + 1;
//     const real_t x69 = x26 * x31;
//     const real_t x70 = x21 * x26;
//     const real_t x71 = x26 * x28;
//     const real_t x72 = x26 * x33;
//     const real_t x73 = u[11] * x69 + u[2] * x72 + u[5] * x70 + u[8] * x71 + 1;
//     const real_t x74 = log(x35 * x56 * x61 - x35 * x66 * x68 + x42 * x47 * x66 - x42 * x56 * x73 -
//                            x47 * x61 * x67 + x67 * x68 * x73);
//     const real_t x75 = u_old[0] * x26;
//     const real_t x76 = u_old[3] * x26;
//     const real_t x77 = u_old[6] * x26;
//     const real_t x78 = u_old[9] * x26;
//     const real_t x79 = -x21 * x76 - x28 * x77 - x31 * x78 - x33 * x75 + x35;
//     const real_t x80 = 1.0 / dt;
//     const real_t x81 = x35 * x80;
//     const real_t x82 = -u_old[10] * x69 - u_old[1] * x72 - u_old[4] * x70 - u_old[7] * x71 + x47;
//     const real_t x83 = x47 * x80;
//     const real_t x84 =
//         x80 * (-u_old[11] * x69 - u_old[2] * x72 - u_old[5] * x70 - u_old[8] * x71 + x73);
//     const real_t x85 = -x36 * x76 - x39 * x77 - x40 * x78 - x41 * x75 + x42;
//     const real_t x86 = x42 * x80;
//     const real_t x87 = -u_old[11] * x57 - u_old[2] * x60 - u_old[5] * x58 - u_old[8] * x59 + x61;
//     const real_t x88 = x61 * x80;
//     const real_t x89 = -u_old[10] * x57 - u_old[1] * x60 - u_old[4] * x58 - u_old[7] * x59 + x68;
//     const real_t x90 = x80 * x89;
//     const real_t x91 = -u_old[10] * x62 - u_old[1] * x65 - u_old[4] * x63 - u_old[7] * x64 + x56;
//     const real_t x92 = x80 * x91;
//     const real_t x93 = -u_old[11] * x62 - u_old[2] * x65 - u_old[5] * x63 - u_old[8] * x64 + x66;
//     const real_t x94 = x66 * x80;
//     const real_t x95 = -x49 * x78 - x51 * x76 - x54 * x77 - x55 * x75 + x67;
//     const real_t x96 = x67 * x80;
//     const real_t x97 = x79 * x80;
//     const real_t x98 = x80 * x82;
//     const real_t x99 = x73 * x80;
//     element_energy =
//         (eta * (2 * pow(x56 * x92 + x93 * x94 + x95 * x96, 2) +
//                 2 * pow(x68 * x90 + x85 * x86 + x87 * x88, 2) +
//                 2 * pow(x73 * x84 + x79 * x81 + x82 * x83, 2) +
//                 pow(x42 * x97 + x61 * x84 + x68 * x98 + x81 * x85 + x83 * x89 + x87 * x99, 2) +
//                 pow(x56 * x90 + x68 * x92 + x85 * x96 + x86 * x95 + x87 * x94 + x88 * x93, 2) +
//                 pow(x56 * x98 + x66 * x84 + x67 * x97 + x81 * x95 + x83 * x91 + x93 * x99, 2)) +
//          (1.0 / 2.0) * lambda * pow(x74, 2) - mu * x74 +
//          (1.0 / 2.0) * mu *
//              (pow(x35, 2) + pow(x42, 2) + pow(x47, 2) + pow(x56, 2) + pow(x61, 2) + pow(x66, 2) +
//               pow(x67, 2) + pow(x68, 2) + pow(x73, 2) - 3)) *
//         (-x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 - x15 * x17 * x3 -
//          x5 * x7 * x9);
// }

// static SFEM_INLINE void viscous_power_density_curnier_gradient(
//     const real_t mu,
//     const real_t lambda,
//     const real_t eta,
//     const real_t px0,
//     const real_t px1,
//     const real_t px2,
//     const real_t px3,
//     const real_t py0,
//     const real_t py1,
//     const real_t py2,
//     const real_t py3,
//     const real_t pz0,
//     const real_t pz1,
//     const real_t pz2,
//     const real_t pz3,
//     const real_t *const SFEM_RESTRICT u_old,
//     const real_t *const SFEM_RESTRICT u,
//     real_t *const SFEM_RESTRICT element_vector) {
//     const real_t x0 = pz0 - pz3;
//     const real_t x1 = -x0;
//     const real_t x2 = py0 - py2;
//     const real_t x3 = -x2;
//     const real_t x4 = px0 - px1;
//     const real_t x5 = -1.0 / 6.0 * x4;
//     const real_t x6 = py0 - py3;
//     const real_t x7 = -x6;
//     const real_t x8 = pz0 - pz2;
//     const real_t x9 = -x8;
//     const real_t x10 = py0 - py1;
//     const real_t x11 = -x10;
//     const real_t x12 = px0 - px2;
//     const real_t x13 = -1.0 / 6.0 * x12;
//     const real_t x14 = pz0 - pz1;
//     const real_t x15 = -x14;
//     const real_t x16 = px0 - px3;
//     const real_t x17 = -1.0 / 6.0 * x16;
//     const real_t x18 = -x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 -
//                        x15 * x17 * x3 - x5 * x7 * x9;
//     const real_t x19 = x2 * x4;
//     const real_t x20 = x12 * x6;
//     const real_t x21 = x10 * x16;
//     const real_t x22 = x4 * x6;
//     const real_t x23 = x10 * x12;
//     const real_t x24 = x16 * x2;
//     const real_t x25 = 1.0 / (x0 * x19 - x0 * x23 + x14 * x20 - x14 * x24 + x21 * x8 - x22 * x8);
//     const real_t x26 = x20 - x24;
//     const real_t x27 = -x25 * x26;
//     const real_t x28 = x25 * (-x21 + x22);
//     const real_t x29 = x19 - x23;
//     const real_t x30 = -x25 * x29;
//     const real_t x31 = x25 * (x21 - x22 + x26 + x29);
//     const real_t x32 = u[0] * x31 + u[3] * x27 + u[6] * x28 + u[9] * x30;
//     const real_t x33 = x10 * x8 - x14 * x2;
//     const real_t x34 = -x25 * x33;
//     const real_t x35 = x0 * x2 - x6 * x8;
//     const real_t x36 = -x25 * x35;
//     const real_t x37 = x0 * x10;
//     const real_t x38 = x14 * x6;
//     const real_t x39 = x25 * (x37 - x38);
//     const real_t x40 = x25 * (x33 + x35 - x37 + x38);
//     const real_t x41 = u[10] * x34 + u[1] * x40 + u[4] * x36 + u[7] * x39;
//     const real_t x42 = -x12 * x14 + x4 * x8;
//     const real_t x43 = x25 * x42;
//     const real_t x44 = x0 * x12 - x16 * x8;
//     const real_t x45 = x25 * x44;
//     const real_t x46 = x0 * x4;
//     const real_t x47 = x14 * x16;
//     const real_t x48 = x25 * (-x46 + x47);
//     const real_t x49 = x25 * (-x42 - x44 + x46 - x47);
//     const real_t x50 = u[11] * x43 + u[2] * x49 + u[5] * x45 + u[8] * x48;
//     const real_t x51 = x41 * x50;
//     const real_t x52 = u[11] * x34 + u[2] * x40 + u[5] * x36 + u[8] * x39;
//     const real_t x53 = u[10] * x43 + u[1] * x49 + u[4] * x45 + u[7] * x48 + 1;
//     const real_t x54 = x52 * x53;
//     const real_t x55 = x51 - x54;
//     const real_t x56 = u[0] * x49 + u[3] * x45 + u[6] * x48 + u[9] * x43;
//     const real_t x57 = u[10] * x30 + u[1] * x31 + u[4] * x27 + u[7] * x28;
//     const real_t x58 = x52 * x57;
//     const real_t x59 = u[11] * x30 + u[2] * x31 + u[5] * x27 + u[8] * x28 + 1;
//     const real_t x60 = x41 * x59;
//     const real_t x61 = u[0] * x40 + u[3] * x36 + u[6] * x39 + u[9] * x34 + 1;
//     const real_t x62 = x50 * x57;
//     const real_t x63 = x32 * x51 - x32 * x54 + x53 * x59 * x61 + x56 * x58 - x56 * x60 - x61 * x62;
//     const real_t x64 = 1.0 / x63;
//     const real_t x65 = mu * x64;
//     const real_t x66 = lambda * x64 * log(x63);
//     const real_t x67 = -u_old[0] * x31 - u_old[3] * x27 - u_old[6] * x28 - u_old[9] * x30 + x32;
//     const real_t x68 = 1.0 / dt;
//     const real_t x69 = x32 * x68;
//     const real_t x70 = -u_old[10] * x30 - u_old[1] * x31 - u_old[4] * x27 - u_old[7] * x28 + x57;
//     const real_t x71 = x57 * x68;
//     const real_t x72 =
//         x68 * (-u_old[11] * x30 - u_old[2] * x31 - u_old[5] * x27 - u_old[8] * x28 + x59);
//     const real_t x73 = 2 * x59 * x72 + 2 * x67 * x69 + 2 * x70 * x71;
//     const real_t x74 = 4 * x32;
//     const real_t x75 = -u_old[0] * x49 - u_old[3] * x45 - u_old[6] * x48 - u_old[9] * x43 + x56;
//     const real_t x76 = x67 * x68;
//     const real_t x77 = -u_old[10] * x43 - u_old[1] * x49 - u_old[4] * x45 - u_old[7] * x48 + x53;
//     const real_t x78 = x68 * x70;
//     const real_t x79 = -u_old[11] * x43 - u_old[2] * x49 - u_old[5] * x45 - u_old[8] * x48 + x50;
//     const real_t x80 = x59 * x68;
//     const real_t x81 = x50 * x72 + x53 * x78 + x56 * x76 + x69 * x75 + x71 * x77 + x79 * x80;
//     const real_t x82 = 4 * x56;
//     const real_t x83 = -u_old[10] * x34 - u_old[1] * x40 - u_old[4] * x36 - u_old[7] * x39 + x41;
//     const real_t x84 = -u_old[0] * x40 - u_old[3] * x36 - u_old[6] * x39 - u_old[9] * x34 + x61;
//     const real_t x85 = -u_old[11] * x34 - u_old[2] * x40 - u_old[5] * x36 - u_old[8] * x39 + x52;
//     const real_t x86 = x41 * x78 + x52 * x72 + x61 * x76 + x69 * x84 + x71 * x83 + x80 * x85;
//     const real_t x87 = 4 * x61;
//     const real_t x88 = (1.0 / 2.0) * eta;
//     const real_t x89 = mu * x32 - x55 * x65 + x55 * x66 + x88 * (x73 * x74 + x81 * x82 + x86 * x87);
//     const real_t x90 = x58 - x60;
//     const real_t x91 = x56 * x68;
//     const real_t x92 = x50 * x68;
//     const real_t x93 = x68 * x77;
//     const real_t x94 = 2 * x53 * x93 + 2 * x75 * x91 + 2 * x79 * x92;
//     const real_t x95 = x52 * x68;
//     const real_t x96 = x61 * x68;
//     const real_t x97 = x68 * x83;
//     const real_t x98 = x41 * x93 + x53 * x97 + x75 * x96 + x79 * x95 + x84 * x91 + x85 * x92;
//     const real_t x99 = mu * x56 - x65 * x90 + x66 * x90 + x88 * (x74 * x81 + x82 * x94 + x87 * x98);
//     const real_t x100 = x53 * x59 - x62;
//     const real_t x101 = 2 * x41 * x97 + 2 * x84 * x96 + 2 * x85 * x95;
//     const real_t x102 =
//         mu * x61 - x100 * x65 + x100 * x66 + x88 * (x101 * x87 + x74 * x86 + x82 * x98);
//     const real_t x103 = -x50 * x61 + x52 * x56;
//     const real_t x104 = 4 * x57;
//     const real_t x105 = 4 * x41;
//     const real_t x106 = 4 * x53;
//     const real_t x107 =
//         mu * x57 - x103 * x65 + x103 * x66 + x88 * (x104 * x73 + x105 * x86 + x106 * x81);
//     const real_t x108 = x32 * x50 - x56 * x59;
//     const real_t x109 =
//         mu * x41 - x108 * x65 + x108 * x66 + x88 * (x101 * x105 + x104 * x86 + x106 * x98);
//     const real_t x110 = -x32 * x52 + x59 * x61;
//     const real_t x111 =
//         mu * x53 - x110 * x65 + x110 * x66 + x88 * (x104 * x81 + x105 * x98 + x106 * x94);
//     const real_t x112 = x32 * x41 - x57 * x61;
//     const real_t x113 = 4 * x50;
//     const real_t x114 = 4 * x52;
//     const real_t x115 = 4 * x59;
//     const real_t x116 =
//         mu * x50 - x112 * x65 + x112 * x66 + x88 * (x113 * x94 + x114 * x98 + x115 * x81);
//     const real_t x117 = -x32 * x53 + x56 * x57;
//     const real_t x118 =
//         mu * x52 - x117 * x65 + x117 * x66 + x88 * (x101 * x114 + x113 * x98 + x115 * x86);
//     const real_t x119 = -x41 * x56 + x53 * x61;
//     const real_t x120 =
//         mu * x59 - x119 * x65 + x119 * x66 + x88 * (x113 * x81 + x114 * x86 + x115 * x73);
//     element_vector[0] = x18 * (x102 * x40 + x31 * x89 + x49 * x99);
//     element_vector[1] = x18 * (x107 * x31 + x109 * x40 + x111 * x49);
//     element_vector[2] = x18 * (x116 * x49 + x118 * x40 + x120 * x31);
//     element_vector[3] = x18 * (x102 * x36 + x27 * x89 + x45 * x99);
//     element_vector[4] = x18 * (x107 * x27 + x109 * x36 + x111 * x45);
//     element_vector[5] = x18 * (x116 * x45 + x118 * x36 + x120 * x27);
//     element_vector[6] = x18 * (x102 * x39 + x28 * x89 + x48 * x99);
//     element_vector[7] = x18 * (x107 * x28 + x109 * x39 + x111 * x48);
//     element_vector[8] = x18 * (x116 * x48 + x118 * x39 + x120 * x28);
//     element_vector[9] = x18 * (x102 * x34 + x30 * x89 + x43 * x99);
//     element_vector[10] = x18 * (x107 * x30 + x109 * x34 + x111 * x43);
//     element_vector[11] = x18 * (x116 * x43 + x118 * x34 + x120 * x30);
// }

// static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
//     int i;
//     for (i = 0; i < size - 4; i += 4) {
//         if (arr[i] == target) return i;
//         if (arr[i + 1] == target) return i + 1;
//         if (arr[i + 2] == target) return i + 2;
//         if (arr[i + 3] == target) return i + 3;
//     }
//     for (; i < size; i++) {
//         if (arr[i] == target) return i;
//     }
//     return -1;
// }

// static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
//     if (lenrow <= 32) {
//         return linear_search(key, row, lenrow);

//         // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
//         // while (key > row[++k]) {
//         //     // Hi
//         // }
//         // assert(k < lenrow);
//         // assert(key == row[k]);
//     } else {
//         // Use this for larger number of dofs per row
//         return find_idx_binary_search(key, row, lenrow);
//     }
// }

// static SFEM_INLINE void find_cols4(const idx_t *targets,
//                                    const idx_t *const row,
//                                    const int lenrow,
//                                    int *ks) {
//     if (lenrow > 32) {
//         for (int d = 0; d < 4; ++d) {
//             ks[d] = find_col(targets[d], row, lenrow);
//         }
//     } else {
// #pragma unroll(4)
//         for (int d = 0; d < 4; ++d) {
//             ks[d] = 0;
//         }

//         for (int i = 0; i < lenrow; ++i) {
// #pragma unroll(4)
//             for (int d = 0; d < 4; ++d) {
//                 ks[d] += row[i] < targets[d];
//             }
//         }
//     }
// }

// static SFEM_INLINE void viscous_power_density_curnier_hessian(
//     const real_t mu,
//     const real_t lambda,
//     const real_t eta,
//     const real_t px0,
//     const real_t px1,
//     const real_t px2,
//     const real_t px3,
//     const real_t py0,
//     const real_t py1,
//     const real_t py2,
//     const real_t py3,
//     const real_t pz0,
//     const real_t pz1,
//     const real_t pz2,
//     const real_t pz3,
//     const real_t *const SFEM_RESTRICT u_old,
//     const real_t *const SFEM_RESTRICT u,
//     real_t *const SFEM_RESTRICT element_matrix) {
//     const real_t x0 = pz0 - pz3;
//     const real_t x1 = -x0;
//     const real_t x2 = py0 - py2;
//     const real_t x3 = -x2;
//     const real_t x4 = px0 - px1;
//     const real_t x5 = -1.0 / 6.0 * x4;
//     const real_t x6 = py0 - py3;
//     const real_t x7 = -x6;
//     const real_t x8 = pz0 - pz2;
//     const real_t x9 = -x8;
//     const real_t x10 = py0 - py1;
//     const real_t x11 = -x10;
//     const real_t x12 = px0 - px2;
//     const real_t x13 = -1.0 / 6.0 * x12;
//     const real_t x14 = pz0 - pz1;
//     const real_t x15 = -x14;
//     const real_t x16 = px0 - px3;
//     const real_t x17 = -1.0 / 6.0 * x16;
//     const real_t x18 = -x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 -
//                        x15 * x17 * x3 - x5 * x7 * x9;
//     const real_t x19 = x2 * x4;
//     const real_t x20 = x12 * x6;
//     const real_t x21 = x10 * x16;
//     const real_t x22 = x4 * x6;
//     const real_t x23 = x10 * x12;
//     const real_t x24 = x16 * x2;
//     const real_t x25 = 1.0 / (x0 * x19 - x0 * x23 + x14 * x20 - x14 * x24 + x21 * x8 - x22 * x8);
//     const real_t x26 = x20 - x24;
//     const real_t x27 = -x25 * x26;
//     const real_t x28 = x25 * (-x21 + x22);
//     const real_t x29 = x19 - x23;
//     const real_t x30 = -x25 * x29;
//     const real_t x31 = x25 * (x21 - x22 + x26 + x29);
//     const real_t x32 = u[0] * x31 + u[3] * x27 + u[6] * x28 + u[9] * x30;
//     const real_t x33 = 1.0 / dt;
//     const real_t x34 = x32 * x33;
//     const real_t x35 = -u_old[0] * x31 - u_old[3] * x27 - u_old[6] * x28 - u_old[9] * x30 + x32;
//     const real_t x36 = x33 * x35;
//     const real_t x37 = x34 + x36;
//     const real_t x38 = 4 * x37;
//     const real_t x39 = x32 * x38;
//     const real_t x40 = x0 * x2 - x6 * x8;
//     const real_t x41 = -x25 * x40;
//     const real_t x42 = x0 * x10;
//     const real_t x43 = x14 * x6;
//     const real_t x44 = x25 * (x42 - x43);
//     const real_t x45 = x10 * x8 - x14 * x2;
//     const real_t x46 = -x25 * x45;
//     const real_t x47 = x25 * (x40 - x42 + x43 + x45);
//     const real_t x48 = u[0] * x47 + u[3] * x41 + u[6] * x44 + u[9] * x46 + 1;
//     const real_t x49 = x33 * x48;
//     const real_t x50 = -u_old[0] * x47 - u_old[3] * x41 - u_old[6] * x44 - u_old[9] * x46 + x48;
//     const real_t x51 = x33 * x50;
//     const real_t x52 = x49 + x51;
//     const real_t x53 = 4 * x48;
//     const real_t x54 = x52 * x53;
//     const real_t x55 = x0 * x12 - x16 * x8;
//     const real_t x56 = x25 * x55;
//     const real_t x57 = x0 * x4;
//     const real_t x58 = x14 * x16;
//     const real_t x59 = x25 * (-x57 + x58);
//     const real_t x60 = -x12 * x14 + x4 * x8;
//     const real_t x61 = x25 * x60;
//     const real_t x62 = x25 * (-x55 + x57 - x58 - x60);
//     const real_t x63 = u[0] * x62 + u[3] * x56 + u[6] * x59 + u[9] * x61;
//     const real_t x64 = x33 * x63;
//     const real_t x65 = -u_old[0] * x62 - u_old[3] * x56 - u_old[6] * x59 - u_old[9] * x61 + x63;
//     const real_t x66 = x33 * x65;
//     const real_t x67 = 2 * x64 + 2 * x66;
//     const real_t x68 = 4 * x63;
//     const real_t x69 = u[11] * x61 + u[2] * x62 + u[5] * x56 + u[8] * x59;
//     const real_t x70 = -u_old[11] * x61 - u_old[2] * x62 - u_old[5] * x56 - u_old[8] * x59 + x69;
//     const real_t x71 = x33 * x69;
//     const real_t x72 = u[10] * x61 + u[1] * x62 + u[4] * x56 + u[7] * x59 + 1;
//     const real_t x73 = -u_old[10] * x61 - u_old[1] * x62 - u_old[4] * x56 - u_old[7] * x59 + x72;
//     const real_t x74 = x33 * x72;
//     const real_t x75 = 8 * x64 * x65 + 8 * x70 * x71 + 8 * x73 * x74;
//     const real_t x76 = (1.0 / 2.0) * eta;
//     const real_t x77 = u[10] * x30 + u[1] * x31 + u[4] * x27 + u[7] * x28;
//     const real_t x78 = u[11] * x46 + u[2] * x47 + u[5] * x41 + u[8] * x44;
//     const real_t x79 = x77 * x78;
//     const real_t x80 = u[10] * x46 + u[1] * x47 + u[4] * x41 + u[7] * x44;
//     const real_t x81 = u[11] * x30 + u[2] * x31 + u[5] * x27 + u[8] * x28 + 1;
//     const real_t x82 = x80 * x81;
//     const real_t x83 = x79 - x82;
//     const real_t x84 = x69 * x80;
//     const real_t x85 = x72 * x78;
//     const real_t x86 = x69 * x77;
//     const real_t x87 = x32 * x84 - x32 * x85 + x48 * x72 * x81 - x48 * x86 + x63 * x79 - x63 * x82;
//     const real_t x88 = pow(x87, -2);
//     const real_t x89 = lambda * x88;
//     const real_t x90 = -x83;
//     const real_t x91 = mu * x88;
//     const real_t x92 = x90 * x91;
//     const real_t x93 = x83 * x89;
//     const real_t x94 = log(x87);
//     const real_t x95 = x90 * x94;
//     const real_t x96 =
//         mu + x76 * (x39 + x54 + x67 * x68 + x75) + pow(x83, 2) * x89 - x83 * x92 + x93 * x95;
//     const real_t x97 = 4 * x65;
//     const real_t x98 = x33 * x77;
//     const real_t x99 = 4 * x73;
//     const real_t x100 = -u_old[11] * x30 - u_old[2] * x31 - u_old[5] * x27 - u_old[8] * x28 + x81;
//     const real_t x101 = 4 * x71;
//     const real_t x102 = -u_old[10] * x30 - u_old[1] * x31 - u_old[4] * x27 - u_old[7] * x28 + x77;
//     const real_t x103 = x102 * x33;
//     const real_t x104 = 4 * x72;
//     const real_t x105 = x33 * x70;
//     const real_t x106 = 4 * x81;
//     const real_t x107 = x100 * x101 + x103 * x104 + x105 * x106 + x34 * x97 + x36 * x68 + x98 * x99;
//     const real_t x108 = x84 - x85;
//     const real_t x109 = x108 * x93;
//     const real_t x110 = x108 * x90;
//     const real_t x111 = x89 * x94;
//     const real_t x112 = x109 + x110 * x111 - x110 * x91 + x76 * (x107 + x37 * x68);
//     const real_t x113 = x33 * x80;
//     const real_t x114 = -u_old[11] * x46 - u_old[2] * x47 - u_old[5] * x41 - u_old[8] * x44 + x78;
//     const real_t x115 = 4 * x78;
//     const real_t x116 = -u_old[10] * x46 - u_old[1] * x47 - u_old[4] * x41 - u_old[7] * x44 + x80;
//     const real_t x117 = x116 * x33;
//     const real_t x118 =
//         x101 * x114 + x104 * x117 + x105 * x115 + x113 * x99 + x49 * x97 + x51 * x68;
//     const real_t x119 = -x72 * x81 + x86;
//     const real_t x120 = -x119;
//     const real_t x121 = x120 * x93;
//     const real_t x122 = x120 * x89;
//     const real_t x123 = -x120 * x92 + x121 + x122 * x95 + x76 * (x118 + x52 * x68);
//     const real_t x124 = x112 * x31 + x123 * x47 + x62 * x96;
//     const real_t x125 = x64 + x66;
//     const real_t x126 = x125 * x68;
//     const real_t x127 = 2 * x34 + 2 * x36;
//     const real_t x128 = 4 * x32;
//     const real_t x129 = x33 * x81;
//     const real_t x130 = 8 * x100 * x129 + 8 * x102 * x98 + 8 * x34 * x35;
//     const real_t x131 = -x108;
//     const real_t x132 = x131 * x91;
//     const real_t x133 = x108 * x89;
//     const real_t x134 = x131 * x94;
//     const real_t x135 = mu + pow(x108, 2) * x89 - x108 * x132 + x133 * x134 +
//                         x76 * (x126 + x127 * x128 + x130 + x54);
//     const real_t x136 = x109 - x132 * x83 + x134 * x93 + x76 * (x107 + x125 * x128);
//     const real_t x137 = 4 * x80;
//     const real_t x138 = x33 * x78;
//     const real_t x139 = x114 * x33;
//     const real_t x140 =
//         4 * x100 * x138 + x103 * x137 + x106 * x139 + 4 * x116 * x98 + 4 * x34 * x50 + x36 * x53;
//     const real_t x141 = x120 * x133;
//     const real_t x142 = -x120 * x132 + x122 * x134 + x141 + x76 * (x128 * x52 + x140);
//     const real_t x143 = x135 * x31 + x136 * x62 + x142 * x47;
//     const real_t x144 = 2 * x49 + 2 * x51;
//     const real_t x145 = 8 * x113 * x116 + 8 * x114 * x138 + 8 * x49 * x50;
//     const real_t x146 = x119 * x91;
//     const real_t x147 = x119 * x94;
//     const real_t x148 = mu + pow(x120, 2) * x89 - x120 * x146 + x122 * x147 +
//                         x76 * (x126 + x144 * x53 + x145 + x39);
//     const real_t x149 = -x108 * x146 + x133 * x147 + x141 + x76 * (x140 + x38 * x48);
//     const real_t x150 = x121 - x146 * x83 + x147 * x93 + x76 * (x118 + x125 * x53);
//     const real_t x151 = x148 * x47 + x149 * x31 + x150 * x62;
//     const real_t x152 = x113 + x117;
//     const real_t x153 = x152 * x53;
//     const real_t x154 = 2 * x103 + 2 * x98;
//     const real_t x155 = x33 * x73;
//     const real_t x156 = x155 + x74;
//     const real_t x157 = x156 * x68;
//     const real_t x158 = -x48 * x69 + x63 * x78;
//     const real_t x159 = x133 * x158;
//     const real_t x160 = -x158;
//     const real_t x161 = x160 * x91;
//     const real_t x162 = x160 * x94;
//     const real_t x163 = -x108 * x161 + x133 * x162 + x159 + x76 * (x128 * x154 + x153 + x157);
//     const real_t x164 = 2 * eta;
//     const real_t x165 = x164 * x32;
//     const real_t x166 = 1.0 / x87;
//     const real_t x167 = mu * x166;
//     const real_t x168 = x167 * x78;
//     const real_t x169 = lambda * x166 * x94;
//     const real_t x170 = x169 * x78;
//     const real_t x171 = x158 * x93 - x168 + x170;
//     const real_t x172 = x156 * x165 - x161 * x83 + x162 * x93 + x171;
//     const real_t x173 = x167 * x69;
//     const real_t x174 = x169 * x69;
//     const real_t x175 = x122 * x158 + x173 - x174;
//     const real_t x176 = -x120 * x161 + x122 * x162 + x152 * x165 + x175;
//     const real_t x177 = x163 * x31 + x172 * x62 + x176 * x47;
//     const real_t x178 = x103 + x98;
//     const real_t x179 = x128 * x178;
//     const real_t x180 = 2 * x113 + 2 * x117;
//     const real_t x181 = x32 * x69 - x63 * x81;
//     const real_t x182 = x122 * x181;
//     const real_t x183 = -x181;
//     const real_t x184 = x183 * x91;
//     const real_t x185 = x183 * x94;
//     const real_t x186 = -x120 * x184 + x122 * x185 + x182 + x76 * (x157 + x179 + x180 * x53);
//     const real_t x187 = x164 * x48;
//     const real_t x188 = x133 * x181 - x173 + x174;
//     const real_t x189 = -x108 * x184 + x133 * x185 + x178 * x187 + x188;
//     const real_t x190 = x167 * x81;
//     const real_t x191 = x169 * x81;
//     const real_t x192 = x181 * x93 + x190 - x191;
//     const real_t x193 = x156 * x187 - x184 * x83 + x185 * x93 + x192;
//     const real_t x194 = x186 * x47 + x189 * x31 + x193 * x62;
//     const real_t x195 = 2 * x155 + 2 * x74;
//     const real_t x196 = x32 * x78 - x48 * x81;
//     const real_t x197 = -x196;
//     const real_t x198 = x197 * x93;
//     const real_t x199 = x196 * x91;
//     const real_t x200 = x196 * x94;
//     const real_t x201 = x198 - x199 * x83 + x200 * x93 + x76 * (x153 + x179 + x195 * x68);
//     const real_t x202 = x164 * x63;
//     const real_t x203 = x133 * x197 + x168 - x170;
//     const real_t x204 = -x108 * x199 + x133 * x200 + x178 * x202 + x203;
//     const real_t x205 = x122 * x197 - x190 + x191;
//     const real_t x206 = -x120 * x199 + x122 * x200 + x152 * x202 + x205;
//     const real_t x207 = x201 * x62 + x204 * x31 + x206 * x47;
//     const real_t x208 = x18 * (x177 * x31 + x194 * x47 + x207 * x62);
//     const real_t x209 = x138 + x139;
//     const real_t x210 = x209 * x53;
//     const real_t x211 = 2 * x105 + 2 * x71;
//     const real_t x212 = x100 * x33;
//     const real_t x213 = x129 + x212;
//     const real_t x214 = x128 * x213;
//     const real_t x215 = x32 * x80 - x48 * x77;
//     const real_t x216 = x215 * x93;
//     const real_t x217 = -x215;
//     const real_t x218 = x217 * x91;
//     const real_t x219 = x217 * x94;
//     const real_t x220 = x216 - x218 * x83 + x219 * x93 + x76 * (x210 + x211 * x68 + x214);
//     const real_t x221 = x167 * x80;
//     const real_t x222 = x169 * x80;
//     const real_t x223 = x133 * x215 - x221 + x222;
//     const real_t x224 = -x108 * x218 + x133 * x219 + x202 * x213 + x223;
//     const real_t x225 = x167 * x77;
//     const real_t x226 = x169 * x77;
//     const real_t x227 = x122 * x215 + x225 - x226;
//     const real_t x228 = -x120 * x218 + x122 * x219 + x202 * x209 + x227;
//     const real_t x229 = x220 * x62 + x224 * x31 + x228 * x47;
//     const real_t x230 = x105 + x71;
//     const real_t x231 = x230 * x68;
//     const real_t x232 = 2 * x138 + 2 * x139;
//     const real_t x233 = x32 * x72 - x63 * x77;
//     const real_t x234 = -x233;
//     const real_t x235 = x122 * x234;
//     const real_t x236 = x233 * x91;
//     const real_t x237 = x233 * x94;
//     const real_t x238 = -x120 * x236 + x122 * x237 + x235 + x76 * (x214 + x231 + x232 * x53);
//     const real_t x239 = -x225 + x226 + x234 * x93;
//     const real_t x240 = x187 * x230 - x236 * x83 + x237 * x93 + x239;
//     const real_t x241 = x167 * x72;
//     const real_t x242 = x169 * x72;
//     const real_t x243 = x133 * x234 + x241 - x242;
//     const real_t x244 = -x108 * x236 + x133 * x237 + x187 * x213 + x243;
//     const real_t x245 = x238 * x47 + x240 * x62 + x244 * x31;
//     const real_t x246 = 2 * x129 + 2 * x212;
//     const real_t x247 = -x48 * x72 + x63 * x80;
//     const real_t x248 = -x247;
//     const real_t x249 = x133 * x248;
//     const real_t x250 = x247 * x91;
//     const real_t x251 = x247 * x94;
//     const real_t x252 = -x108 * x250 + x133 * x251 + x249 + x76 * (x128 * x246 + x210 + x231);
//     const real_t x253 = x221 - x222 + x248 * x93;
//     const real_t x254 = x165 * x230 - x250 * x83 + x251 * x93 + x253;
//     const real_t x255 = x122 * x248 - x241 + x242;
//     const real_t x256 = -x120 * x250 + x122 * x251 + x165 * x209 + x255;
//     const real_t x257 = x252 * x31 + x254 * x62 + x256 * x47;
//     const real_t x258 = x18 * (x229 * x62 + x245 * x47 + x257 * x31);
//     const real_t x259 = x18 * (x124 * x56 + x143 * x27 + x151 * x41);
//     const real_t x260 = x18 * (x177 * x27 + x194 * x41 + x207 * x56);
//     const real_t x261 = x18 * (x229 * x56 + x245 * x41 + x257 * x27);
//     const real_t x262 = x18 * (x124 * x59 + x143 * x28 + x151 * x44);
//     const real_t x263 = x18 * (x177 * x28 + x194 * x44 + x207 * x59);
//     const real_t x264 = x18 * (x229 * x59 + x245 * x44 + x257 * x28);
//     const real_t x265 = x18 * (x124 * x61 + x143 * x30 + x151 * x46);
//     const real_t x266 = x18 * (x177 * x30 + x194 * x46 + x207 * x61);
//     const real_t x267 = x18 * (x229 * x61 + x245 * x46 + x257 * x30);
//     const real_t x268 = 4 * x178;
//     const real_t x269 = x268 * x77;
//     const real_t x270 = x104 * x156;
//     const real_t x271 = x181 * x89;
//     const real_t x272 = mu + pow(x181, 2) * x89 - x181 * x184 + x185 * x271 +
//                         x76 * (x137 * x180 + x145 + x269 + x270);
//     const real_t x273 = x158 * x271;
//     const real_t x274 = x158 * x89;
//     const real_t x275 = -x158 * x184 + x185 * x274 + x273 + x76 * (x137 * x178 + x140);
//     const real_t x276 = x197 * x271;
//     const real_t x277 = x197 * x89;
//     const real_t x278 = -x184 * x197 + x185 * x277 + x276 + x76 * (x118 + x137 * x156);
//     const real_t x279 = x272 * x47 + x275 * x31 + x278 * x62;
//     const real_t x280 = x137 * x152;
//     const real_t x281 = 4 * x77;
//     const real_t x282 = mu + pow(x158, 2) * x89 - x158 * x161 + x162 * x274 +
//                         x76 * (x130 + x154 * x281 + x270 + x280);
//     const real_t x283 = -x161 * x181 + x162 * x271 + x273 + x76 * (x140 + x152 * x281);
//     const real_t x284 = x197 * x274;
//     const real_t x285 = -x161 * x197 + x162 * x277 + x284 + x76 * (x107 + x156 * x281);
//     const real_t x286 = x282 * x31 + x283 * x47 + x285 * x62;
//     const real_t x287 = mu + pow(x197, 2) * x89 - x197 * x199 + x200 * x277 +
//                         x76 * (x104 * x195 + x269 + x280 + x75);
//     const real_t x288 = -x158 * x199 + x200 * x274 + x284 + x76 * (x107 + x268 * x72);
//     const real_t x289 = -x181 * x199 + x200 * x271 + x276 + x76 * (x104 * x152 + x118);
//     const real_t x290 = x287 * x62 + x288 * x31 + x289 * x47;
//     const real_t x291 = x104 * x230;
//     const real_t x292 = x213 * x281;
//     const real_t x293 = x234 * x271;
//     const real_t x294 = -x181 * x236 + x237 * x271 + x293 + x76 * (x137 * x232 + x291 + x292);
//     const real_t x295 = x164 * x80;
//     const real_t x296 = x167 * x63;
//     const real_t x297 = x169 * x63;
//     const real_t x298 = x234 * x274 - x296 + x297;
//     const real_t x299 = -x158 * x236 + x213 * x295 + x237 * x274 + x298;
//     const real_t x300 = x167 * x32;
//     const real_t x301 = x169 * x32;
//     const real_t x302 = x234 * x277 + x300 - x301;
//     const real_t x303 = -x197 * x236 + x230 * x295 + x237 * x277 + x302;
//     const real_t x304 = x294 * x47 + x299 * x31 + x303 * x62;
//     const real_t x305 = x137 * x209;
//     const real_t x306 = x215 * x277;
//     const real_t x307 = -x197 * x218 + x219 * x277 + x306 + x76 * (x104 * x211 + x292 + x305);
//     const real_t x308 = x164 * x72;
//     const real_t x309 = x215 * x271 - x300 + x301;
//     const real_t x310 = -x181 * x218 + x209 * x308 + x219 * x271 + x309;
//     const real_t x311 = x167 * x48;
//     const real_t x312 = x169 * x48;
//     const real_t x313 = x215 * x274 + x311 - x312;
//     const real_t x314 = -x158 * x218 + x213 * x308 + x219 * x274 + x313;
//     const real_t x315 = x307 * x62 + x31 * x314 + x310 * x47;
//     const real_t x316 = x248 * x274;
//     const real_t x317 = -x158 * x250 + x251 * x274 + x316 + x76 * (x246 * x281 + x291 + x305);
//     const real_t x318 = x164 * x77;
//     const real_t x319 = x248 * x271 + x296 - x297;
//     const real_t x320 = -x181 * x250 + x209 * x318 + x251 * x271 + x319;
//     const real_t x321 = x248 * x277 - x311 + x312;
//     const real_t x322 = -x197 * x250 + x230 * x318 + x251 * x277 + x321;
//     const real_t x323 = x31 * x317 + x320 * x47 + x322 * x62;
//     const real_t x324 = x18 * (x304 * x47 + x31 * x323 + x315 * x62);
//     const real_t x325 = x104 * x125;
//     const real_t x326 = x137 * x52;
//     const real_t x327 = -x132 * x158 + x134 * x274 + x159 + x76 * (x127 * x281 + x325 + x326);
//     const real_t x328 = -x132 * x181 + x134 * x271 + x188 + x318 * x52;
//     const real_t x329 = x125 * x318 - x132 * x197 + x134 * x277 + x203;
//     const real_t x330 = x31 * x327 + x328 * x47 + x329 * x62;
//     const real_t x331 = x38 * x77;
//     const real_t x332 = -x197 * x92 + x198 + x277 * x95 + x76 * (x104 * x67 + x326 + x331);
//     const real_t x333 = -x158 * x92 + x171 + x274 * x95 + x308 * x37;
//     const real_t x334 = -x181 * x92 + x192 + x271 * x95 + x308 * x52;
//     const real_t x335 = x31 * x333 + x332 * x62 + x334 * x47;
//     const real_t x336 = -x146 * x181 + x147 * x271 + x182 + x76 * (x137 * x144 + x325 + x331);
//     const real_t x337 = -x146 * x158 + x147 * x274 + x175 + x295 * x37;
//     const real_t x338 = x125 * x295 - x146 * x197 + x147 * x277 + x205;
//     const real_t x339 = x31 * x337 + x336 * x47 + x338 * x62;
//     const real_t x340 = x18 * (x27 * x330 + x335 * x56 + x339 * x41);
//     const real_t x341 = x18 * (x27 * x286 + x279 * x41 + x290 * x56);
//     const real_t x342 = x18 * (x27 * x323 + x304 * x41 + x315 * x56);
//     const real_t x343 = x18 * (x28 * x330 + x335 * x59 + x339 * x44);
//     const real_t x344 = x18 * (x279 * x44 + x28 * x286 + x290 * x59);
//     const real_t x345 = x18 * (x28 * x323 + x304 * x44 + x315 * x59);
//     const real_t x346 = x18 * (x30 * x330 + x335 * x61 + x339 * x46);
//     const real_t x347 = x18 * (x279 * x46 + x286 * x30 + x290 * x61);
//     const real_t x348 = x18 * (x30 * x323 + x304 * x46 + x315 * x61);
//     const real_t x349 = 4 * x230;
//     const real_t x350 = x349 * x69;
//     const real_t x351 = x106 * x213;
//     const real_t x352 = x234 * x89;
//     const real_t x353 = mu + pow(x234, 2) * x89 - x234 * x236 + x237 * x352 +
//                         x76 * (x115 * x232 + x145 + x350 + x351);
//     const real_t x354 = x215 * x352;
//     const real_t x355 = x215 * x89;
//     const real_t x356 = -x215 * x236 + x237 * x355 + x354 + x76 * (x115 * x230 + x118);
//     const real_t x357 = x248 * x352;
//     const real_t x358 = x111 * x248;
//     const real_t x359 = x233 * x358 - x236 * x248 + x357 + x76 * (x115 * x213 + x140);
//     const real_t x360 = x31 * x359 + x353 * x47 + x356 * x62;
//     const real_t x361 = x115 * x209;
//     const real_t x362 = 4 * x69;
//     const real_t x363 = mu + pow(x215, 2) * x89 - x215 * x218 + x219 * x355 +
//                         x76 * (x211 * x362 + x351 + x361 + x75);
//     const real_t x364 = -x218 * x234 + x219 * x352 + x354 + x76 * (x118 + x209 * x362);
//     const real_t x365 = x248 * x355;
//     const real_t x366 = x217 * x358 - x218 * x248 + x365 + x76 * (x107 + x213 * x362);
//     const real_t x367 = x31 * x366 + x363 * x62 + x364 * x47;
//     const real_t x368 = mu + x247 * x358 + pow(x248, 2) * x89 - x248 * x250 +
//                         x76 * (x106 * x246 + x130 + x350 + x361);
//     const real_t x369 = -x215 * x250 + x251 * x355 + x365 + x76 * (x107 + x349 * x81);
//     const real_t x370 = -x234 * x250 + x251 * x352 + x357 + x76 * (x106 * x209 + x140);
//     const real_t x371 = x31 * x368 + x369 * x62 + x370 * x47;
//     const real_t x372 = x38 * x81;
//     const real_t x373 = x115 * x52;
//     const real_t x374 = -x215 * x92 + x216 + x355 * x95 + x76 * (x362 * x67 + x372 + x373);
//     const real_t x375 = x164 * x69;
//     const real_t x376 = -x234 * x92 + x239 + x352 * x95 + x375 * x52;
//     const real_t x377 = -x248 * x92 + x253 + x358 * x90 + x37 * x375;
//     const real_t x378 = x31 * x377 + x374 * x62 + x376 * x47;
//     const real_t x379 = x125 * x362;
//     const real_t x380 = x131 * x358 - x132 * x248 + x249 + x76 * (x106 * x127 + x373 + x379);
//     const real_t x381 = x164 * x81;
//     const real_t x382 = x125 * x381 - x132 * x215 + x134 * x355 + x223;
//     const real_t x383 = -x132 * x234 + x134 * x352 + x243 + x381 * x52;
//     const real_t x384 = x31 * x380 + x382 * x62 + x383 * x47;
//     const real_t x385 = -x146 * x234 + x147 * x352 + x235 + x76 * (x115 * x144 + x372 + x379);
//     const real_t x386 = x164 * x78;
//     const real_t x387 = x125 * x386 - x146 * x215 + x147 * x355 + x227;
//     const real_t x388 = x119 * x358 - x146 * x248 + x255 + x37 * x386;
//     const real_t x389 = x31 * x388 + x385 * x47 + x387 * x62;
//     const real_t x390 = x18 * (x27 * x384 + x378 * x56 + x389 * x41);
//     const real_t x391 = x268 * x81;
//     const real_t x392 = x156 * x362;
//     const real_t x393 = -x184 * x234 + x185 * x352 + x293 + x76 * (x115 * x180 + x391 + x392);
//     const real_t x394 = x156 * x386 - x184 * x215 + x185 * x355 + x309;
//     const real_t x395 = x178 * x386 + x183 * x358 - x184 * x248 + x319;
//     const real_t x396 = x31 * x395 + x393 * x47 + x394 * x62;
//     const real_t x397 = x115 * x152;
//     const real_t x398 = x160 * x358 - x161 * x248 + x316 + x76 * (x106 * x154 + x392 + x397);
//     const real_t x399 = x152 * x381 - x161 * x234 + x162 * x352 + x298;
//     const real_t x400 = x156 * x381 - x161 * x215 + x162 * x355 + x313;
//     const real_t x401 = x31 * x398 + x399 * x47 + x400 * x62;
//     const real_t x402 = -x199 * x215 + x200 * x355 + x306 + x76 * (x195 * x362 + x391 + x397);
//     const real_t x403 = x152 * x375 - x199 * x234 + x200 * x352 + x302;
//     const real_t x404 = x178 * x375 + x196 * x358 - x199 * x248 + x321;
//     const real_t x405 = x31 * x404 + x402 * x62 + x403 * x47;
//     const real_t x406 = x18 * (x27 * x401 + x396 * x41 + x405 * x56);
//     const real_t x407 = x18 * (x27 * x371 + x360 * x41 + x367 * x56);
//     const real_t x408 = x18 * (x28 * x384 + x378 * x59 + x389 * x44);
//     const real_t x409 = x18 * (x28 * x401 + x396 * x44 + x405 * x59);
//     const real_t x410 = x18 * (x28 * x371 + x360 * x44 + x367 * x59);
//     const real_t x411 = x18 * (x30 * x384 + x378 * x61 + x389 * x46);
//     const real_t x412 = x18 * (x30 * x401 + x396 * x46 + x405 * x61);
//     const real_t x413 = x18 * (x30 * x371 + x360 * x46 + x367 * x61);
//     const real_t x414 = x135 * x27 + x136 * x56 + x142 * x41;
//     const real_t x415 = x112 * x27 + x123 * x41 + x56 * x96;
//     const real_t x416 = x148 * x41 + x149 * x27 + x150 * x56;
//     const real_t x417 = x163 * x27 + x172 * x56 + x176 * x41;
//     const real_t x418 = x186 * x41 + x189 * x27 + x193 * x56;
//     const real_t x419 = x201 * x56 + x204 * x27 + x206 * x41;
//     const real_t x420 = x18 * (x27 * x417 + x41 * x418 + x419 * x56);
//     const real_t x421 = x220 * x56 + x224 * x27 + x228 * x41;
//     const real_t x422 = x238 * x41 + x240 * x56 + x244 * x27;
//     const real_t x423 = x252 * x27 + x254 * x56 + x256 * x41;
//     const real_t x424 = x18 * (x27 * x423 + x41 * x422 + x421 * x56);
//     const real_t x425 = x18 * (x28 * x414 + x415 * x59 + x416 * x44);
//     const real_t x426 = x18 * (x28 * x417 + x418 * x44 + x419 * x59);
//     const real_t x427 = x18 * (x28 * x423 + x421 * x59 + x422 * x44);
//     const real_t x428 = x18 * (x30 * x414 + x415 * x61 + x416 * x46);
//     const real_t x429 = x18 * (x30 * x417 + x418 * x46 + x419 * x61);
//     const real_t x430 = x18 * (x30 * x423 + x421 * x61 + x422 * x46);
//     const real_t x431 = x27 * x282 + x283 * x41 + x285 * x56;
//     const real_t x432 = x27 * x275 + x272 * x41 + x278 * x56;
//     const real_t x433 = x27 * x288 + x287 * x56 + x289 * x41;
//     const real_t x434 = x27 * x299 + x294 * x41 + x303 * x56;
//     const real_t x435 = x27 * x314 + x307 * x56 + x310 * x41;
//     const real_t x436 = x27 * x317 + x320 * x41 + x322 * x56;
//     const real_t x437 = x18 * (x27 * x436 + x41 * x434 + x435 * x56);
//     const real_t x438 = x27 * x327 + x328 * x41 + x329 * x56;
//     const real_t x439 = x27 * x333 + x332 * x56 + x334 * x41;
//     const real_t x440 = x27 * x337 + x336 * x41 + x338 * x56;
//     const real_t x441 = x18 * (x28 * x438 + x439 * x59 + x44 * x440);
//     const real_t x442 = x18 * (x28 * x431 + x432 * x44 + x433 * x59);
//     const real_t x443 = x18 * (x28 * x436 + x434 * x44 + x435 * x59);
//     const real_t x444 = x18 * (x30 * x438 + x439 * x61 + x440 * x46);
//     const real_t x445 = x18 * (x30 * x431 + x432 * x46 + x433 * x61);
//     const real_t x446 = x18 * (x30 * x436 + x434 * x46 + x435 * x61);
//     const real_t x447 = x27 * x366 + x363 * x56 + x364 * x41;
//     const real_t x448 = x27 * x359 + x353 * x41 + x356 * x56;
//     const real_t x449 = x27 * x368 + x369 * x56 + x370 * x41;
//     const real_t x450 = x27 * x377 + x374 * x56 + x376 * x41;
//     const real_t x451 = x27 * x380 + x382 * x56 + x383 * x41;
//     const real_t x452 = x27 * x388 + x385 * x41 + x387 * x56;
//     const real_t x453 = x18 * (x28 * x451 + x44 * x452 + x450 * x59);
//     const real_t x454 = x27 * x395 + x393 * x41 + x394 * x56;
//     const real_t x455 = x27 * x398 + x399 * x41 + x400 * x56;
//     const real_t x456 = x27 * x404 + x402 * x56 + x403 * x41;
//     const real_t x457 = x18 * (x28 * x455 + x44 * x454 + x456 * x59);
//     const real_t x458 = x18 * (x28 * x449 + x44 * x448 + x447 * x59);
//     const real_t x459 = x18 * (x30 * x451 + x450 * x61 + x452 * x46);
//     const real_t x460 = x18 * (x30 * x455 + x454 * x46 + x456 * x61);
//     const real_t x461 = x18 * (x30 * x449 + x447 * x61 + x448 * x46);
//     const real_t x462 = x135 * x28 + x136 * x59 + x142 * x44;
//     const real_t x463 = x112 * x28 + x123 * x44 + x59 * x96;
//     const real_t x464 = x148 * x44 + x149 * x28 + x150 * x59;
//     const real_t x465 = x163 * x28 + x172 * x59 + x176 * x44;
//     const real_t x466 = x186 * x44 + x189 * x28 + x193 * x59;
//     const real_t x467 = x201 * x59 + x204 * x28 + x206 * x44;
//     const real_t x468 = x18 * (x28 * x465 + x44 * x466 + x467 * x59);
//     const real_t x469 = x220 * x59 + x224 * x28 + x228 * x44;
//     const real_t x470 = x238 * x44 + x240 * x59 + x244 * x28;
//     const real_t x471 = x252 * x28 + x254 * x59 + x256 * x44;
//     const real_t x472 = x18 * (x28 * x471 + x44 * x470 + x469 * x59);
//     const real_t x473 = x18 * (x30 * x462 + x46 * x464 + x463 * x61);
//     const real_t x474 = x18 * (x30 * x465 + x46 * x466 + x467 * x61);
//     const real_t x475 = x18 * (x30 * x471 + x46 * x470 + x469 * x61);
//     const real_t x476 = x28 * x282 + x283 * x44 + x285 * x59;
//     const real_t x477 = x272 * x44 + x275 * x28 + x278 * x59;
//     const real_t x478 = x28 * x288 + x287 * x59 + x289 * x44;
//     const real_t x479 = x28 * x299 + x294 * x44 + x303 * x59;
//     const real_t x480 = x28 * x314 + x307 * x59 + x310 * x44;
//     const real_t x481 = x28 * x317 + x320 * x44 + x322 * x59;
//     const real_t x482 = x18 * (x28 * x481 + x44 * x479 + x480 * x59);
//     const real_t x483 = x18 * (x30 * (x28 * x327 + x328 * x44 + x329 * x59) +
//                                x46 * (x28 * x337 + x336 * x44 + x338 * x59) +
//                                x61 * (x28 * x333 + x332 * x59 + x334 * x44));
//     const real_t x484 = x18 * (x30 * x476 + x46 * x477 + x478 * x61);
//     const real_t x485 = x18 * (x30 * x481 + x46 * x479 + x480 * x61);
//     const real_t x486 = x28 * x366 + x363 * x59 + x364 * x44;
//     const real_t x487 = x28 * x359 + x353 * x44 + x356 * x59;
//     const real_t x488 = x28 * x368 + x369 * x59 + x370 * x44;
//     const real_t x489 = x18 * (x30 * (x28 * x380 + x382 * x59 + x383 * x44) +
//                                x46 * (x28 * x388 + x385 * x44 + x387 * x59) +
//                                x61 * (x28 * x377 + x374 * x59 + x376 * x44));
//     const real_t x490 = x18 * (x30 * (x28 * x398 + x399 * x44 + x400 * x59) +
//                                x46 * (x28 * x395 + x393 * x44 + x394 * x59) +
//                                x61 * (x28 * x404 + x402 * x59 + x403 * x44));
//     const real_t x491 = x18 * (x30 * x488 + x46 * x487 + x486 * x61);
//     const real_t x492 = x18 * (x30 * (x163 * x30 + x172 * x61 + x176 * x46) +
//                                x46 * (x186 * x46 + x189 * x30 + x193 * x61) +
//                                x61 * (x201 * x61 + x204 * x30 + x206 * x46));
//     const real_t x493 = x18 * (x30 * (x252 * x30 + x254 * x61 + x256 * x46) +
//                                x46 * (x238 * x46 + x240 * x61 + x244 * x30) +
//                                x61 * (x220 * x61 + x224 * x30 + x228 * x46));
//     const real_t x494 = x18 * (x30 * (x30 * x317 + x320 * x46 + x322 * x61) +
//                                x46 * (x294 * x46 + x299 * x30 + x303 * x61) +
//                                x61 * (x30 * x314 + x307 * x61 + x310 * x46));
//     element_matrix[0] = x18 * (x124 * x62 + x143 * x31 + x151 * x47);
//     element_matrix[1] = x208;
//     element_matrix[2] = x258;
//     element_matrix[3] = x259;
//     element_matrix[4] = x260;
//     element_matrix[5] = x261;
//     element_matrix[6] = x262;
//     element_matrix[7] = x263;
//     element_matrix[8] = x264;
//     element_matrix[9] = x265;
//     element_matrix[10] = x266;
//     element_matrix[11] = x267;
//     element_matrix[12] = x208;
//     element_matrix[13] = x18 * (x279 * x47 + x286 * x31 + x290 * x62);
//     element_matrix[14] = x324;
//     element_matrix[15] = x340;
//     element_matrix[16] = x341;
//     element_matrix[17] = x342;
//     element_matrix[18] = x343;
//     element_matrix[19] = x344;
//     element_matrix[20] = x345;
//     element_matrix[21] = x346;
//     element_matrix[22] = x347;
//     element_matrix[23] = x348;
//     element_matrix[24] = x258;
//     element_matrix[25] = x324;
//     element_matrix[26] = x18 * (x31 * x371 + x360 * x47 + x367 * x62);
//     element_matrix[27] = x390;
//     element_matrix[28] = x406;
//     element_matrix[29] = x407;
//     element_matrix[30] = x408;
//     element_matrix[31] = x409;
//     element_matrix[32] = x410;
//     element_matrix[33] = x411;
//     element_matrix[34] = x412;
//     element_matrix[35] = x413;
//     element_matrix[36] = x259;
//     element_matrix[37] = x340;
//     element_matrix[38] = x390;
//     element_matrix[39] = x18 * (x27 * x414 + x41 * x416 + x415 * x56);
//     element_matrix[40] = x420;
//     element_matrix[41] = x424;
//     element_matrix[42] = x425;
//     element_matrix[43] = x426;
//     element_matrix[44] = x427;
//     element_matrix[45] = x428;
//     element_matrix[46] = x429;
//     element_matrix[47] = x430;
//     element_matrix[48] = x260;
//     element_matrix[49] = x341;
//     element_matrix[50] = x406;
//     element_matrix[51] = x420;
//     element_matrix[52] = x18 * (x27 * x431 + x41 * x432 + x433 * x56);
//     element_matrix[53] = x437;
//     element_matrix[54] = x441;
//     element_matrix[55] = x442;
//     element_matrix[56] = x443;
//     element_matrix[57] = x444;
//     element_matrix[58] = x445;
//     element_matrix[59] = x446;
//     element_matrix[60] = x261;
//     element_matrix[61] = x342;
//     element_matrix[62] = x407;
//     element_matrix[63] = x424;
//     element_matrix[64] = x437;
//     element_matrix[65] = x18 * (x27 * x449 + x41 * x448 + x447 * x56);
//     element_matrix[66] = x453;
//     element_matrix[67] = x457;
//     element_matrix[68] = x458;
//     element_matrix[69] = x459;
//     element_matrix[70] = x460;
//     element_matrix[71] = x461;
//     element_matrix[72] = x262;
//     element_matrix[73] = x343;
//     element_matrix[74] = x408;
//     element_matrix[75] = x425;
//     element_matrix[76] = x441;
//     element_matrix[77] = x453;
//     element_matrix[78] = x18 * (x28 * x462 + x44 * x464 + x463 * x59);
//     element_matrix[79] = x468;
//     element_matrix[80] = x472;
//     element_matrix[81] = x473;
//     element_matrix[82] = x474;
//     element_matrix[83] = x475;
//     element_matrix[84] = x263;
//     element_matrix[85] = x344;
//     element_matrix[86] = x409;
//     element_matrix[87] = x426;
//     element_matrix[88] = x442;
//     element_matrix[89] = x457;
//     element_matrix[90] = x468;
//     element_matrix[91] = x18 * (x28 * x476 + x44 * x477 + x478 * x59);
//     element_matrix[92] = x482;
//     element_matrix[93] = x483;
//     element_matrix[94] = x484;
//     element_matrix[95] = x485;
//     element_matrix[96] = x264;
//     element_matrix[97] = x345;
//     element_matrix[98] = x410;
//     element_matrix[99] = x427;
//     element_matrix[100] = x443;
//     element_matrix[101] = x458;
//     element_matrix[102] = x472;
//     element_matrix[103] = x482;
//     element_matrix[104] = x18 * (x28 * x488 + x44 * x487 + x486 * x59);
//     element_matrix[105] = x489;
//     element_matrix[106] = x490;
//     element_matrix[107] = x491;
//     element_matrix[108] = x265;
//     element_matrix[109] = x346;
//     element_matrix[110] = x411;
//     element_matrix[111] = x428;
//     element_matrix[112] = x444;
//     element_matrix[113] = x459;
//     element_matrix[114] = x473;
//     element_matrix[115] = x483;
//     element_matrix[116] = x489;
//     element_matrix[117] = x18 * (x30 * (x135 * x30 + x136 * x61 + x142 * x46) +
//                                  x46 * (x148 * x46 + x149 * x30 + x150 * x61) +
//                                  x61 * (x112 * x30 + x123 * x46 + x61 * x96));
//     element_matrix[118] = x492;
//     element_matrix[119] = x493;
//     element_matrix[120] = x266;
//     element_matrix[121] = x347;
//     element_matrix[122] = x412;
//     element_matrix[123] = x429;
//     element_matrix[124] = x445;
//     element_matrix[125] = x460;
//     element_matrix[126] = x474;
//     element_matrix[127] = x484;
//     element_matrix[128] = x490;
//     element_matrix[129] = x492;
//     element_matrix[130] = x18 * (x30 * (x282 * x30 + x283 * x46 + x285 * x61) +
//                                  x46 * (x272 * x46 + x275 * x30 + x278 * x61) +
//                                  x61 * (x287 * x61 + x288 * x30 + x289 * x46));
//     element_matrix[131] = x494;
//     element_matrix[132] = x267;
//     element_matrix[133] = x348;
//     element_matrix[134] = x413;
//     element_matrix[135] = x430;
//     element_matrix[136] = x446;
//     element_matrix[137] = x461;
//     element_matrix[138] = x475;
//     element_matrix[139] = x485;
//     element_matrix[140] = x491;
//     element_matrix[141] = x493;
//     element_matrix[142] = x494;
//     element_matrix[143] = x18 * (x30 * (x30 * x368 + x369 * x61 + x370 * x46) +
//                                  x46 * (x30 * x359 + x353 * x46 + x356 * x61) +
//                                  x61 * (x30 * x366 + x363 * x61 + x364 * x46));
// }

// static int check_symmetric(int n, const real_t *const element_matrix) {
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             const real_t diff = element_matrix[i * n + j] - element_matrix[i + j * n];
//             assert(diff < 1e-16);
//             if (diff > 1e-16) {
//                 return 1;
//             }

//             // printf("%g ",  element_matrix[i*n + j] );
//         }

//         // printf("\n");
//     }

//     // printf("\n");

//     return 0;
// }

// static void numerate(int n, real_t *const element_matrix) {
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             element_matrix[i * n + j] = i * n + j;
//         }
//     }
// }

// void viscous_power_density_curnier_assemble_hessian(const ptrdiff_t nelements,
//                                                     const ptrdiff_t nnodes,
//                                                     idx_t *const SFEM_RESTRICT elems[4],
//                                                     geom_t *const SFEM_RESTRICT xyz[3],
//                                                     const real_t mu,
//                                                     const real_t lambda,
//                                                     const real_t eta,
//                                                     const real_t dt,
//                                                     const real_t *const SFEM_RESTRICT
//                                                         displacement_old,
//                                                     const real_t *const SFEM_RESTRICT displacement,
//                                                     count_t *const SFEM_RESTRICT rowptr,
//                                                     idx_t *const SFEM_RESTRICT colidx,
//                                                     real_t *const SFEM_RESTRICT values) {
//     SFEM_UNUSED(nnodes);

//     double tick = MPI_Wtime();

//     static const int block_size = 3;
//     static const int mat_block_size = block_size * block_size;

// #pragma omp parallel
//     {
// #pragma omp for  // nowait
//         for (ptrdiff_t i = 0; i < nelements; ++i) {
//             idx_t ev[4];
//             idx_t ks[4];

//             real_t element_matrix[(4 * 3) * (4 * 3)];
//             real_t element_displacement[(4 * 3)];
// #pragma unroll(4)
//             for (int v = 0; v < 4; ++v) {
//                 ev[v] = elems[v][i];
//             }

//             // Element indices
//             const idx_t i0 = ev[0];
//             const idx_t i1 = ev[1];
//             const idx_t i2 = ev[2];
//             const idx_t i3 = ev[3];

//             for (int edof_i = 0; edof_i < 4; ++edof_i) {
//                 idx_t dof = ev[edof_i] * block_size;

//                 for (int b = 0; b < block_size; ++b) {
//                     // element_displacement[b * 4 + edof_i] = displacement[dof + b];
//                     element_displacement[b + edof_i * block_size] =
//                         displacement[dof + b];  // OLD
//                                                 // Layout
//                 }
//             }

//             viscous_power_density_curnier_hessian(  // Model parameters
//                 mu,
//                 lambda,
//                 // X-coordinates
//                 xyz[0][i0],
//                 xyz[0][i1],
//                 xyz[0][i2],
//                 xyz[0][i3],
//                 // Y-coordinates
//                 xyz[1][i0],
//                 xyz[1][i1],
//                 xyz[1][i2],
//                 xyz[1][i3],
//                 // Z-coordinates
//                 xyz[2][i0],
//                 xyz[2][i1],
//                 xyz[2][i2],
//                 xyz[2][i3],
//                 // element dispalcement
//                 element_displacement,
//                 // output matrix
//                 element_matrix);

//             assert(!check_symmetric(4 * block_size, element_matrix));

//             for (int edof_i = 0; edof_i < 4; ++edof_i) {
//                 const idx_t dof_i = elems[edof_i][i];
//                 const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

//                 {
//                     const idx_t *row = &colidx[rowptr[dof_i]];
//                     find_cols4(ev, row, lenrow, ks);
//                 }

//                 // Blocks for row
//                 real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

//                 for (int edof_j = 0; edof_j < 4; ++edof_j) {
//                     const idx_t offset_j = ks[edof_j] * block_size;

//                     for (int bi = 0; bi < block_size; ++bi) {
//                         // const int ii = bi * 4 + edof_i;
//                         const int ii = edof_i * block_size + bi;

//                         // Jump rows (including the block-size for the columns)
//                         real_t *row = &block_start[bi * lenrow * block_size];

//                         for (int bj = 0; bj < block_size; ++bj) {
//                             // const int jj = bj * 4 + edof_j;
//                             const int jj = edof_j * block_size + bj;

//                             const real_t val = element_matrix[ii * 12 + jj];

// #pragma omp atomic update
//                             row[offset_j + bj] += val;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     double tock = MPI_Wtime();
//     printf(
//         "viscous_power_density_curnier.c: viscous_power_density_curnier_assemble_hessian\t%g "
//         "seconds\n",
//         tock - tick);
// }

// void viscous_power_density_curnier_assemble_gradient(const ptrdiff_t nelements,
//                                                      const ptrdiff_t nnodes,
//                                                      idx_t *const SFEM_RESTRICT elems[4],
//                                                      geom_t *const SFEM_RESTRICT xyz[3],
//                                                      const real_t mu,
//                                                      const real_t lambda,
//                                                      const real_t eta,
//                                                      const real_t dt,
//                                                      const real_t *const SFEM_RESTRICT
//                                                          displacement_old,
//                                                      const real_t *const SFEM_RESTRICT displacement,
//                                                      real_t *const SFEM_RESTRICT values) {
//     SFEM_UNUSED(nnodes);

//     double tick = MPI_Wtime();

//     static const int block_size = 3;

// #pragma omp parallel
//     {
// #pragma omp for  // nowait
//         for (ptrdiff_t i = 0; i < nelements; ++i) {
//             idx_t ev[4];
//             idx_t ks[4];

//             real_t element_vector[(4 * 3)];
//             real_t element_displacement[(4 * 3)];

// #pragma unroll(4)
//             for (int v = 0; v < 4; ++v) {
//                 ev[v] = elems[v][i];
//             }

//             // Element indices
//             const idx_t i0 = ev[0];
//             const idx_t i1 = ev[1];
//             const idx_t i2 = ev[2];
//             const idx_t i3 = ev[3];

//             for (int edof_i = 0; edof_i < 4; ++edof_i) {
//                 idx_t dof = ev[edof_i] * block_size;

//                 for (int b = 0; b < block_size; ++b) {
//                     // element_displacement[b * 4 + edof_i] = displacement[dof + b];
//                     element_displacement[b + edof_i * block_size] =
//                         displacement[dof + b];  // OLD
//                                                 // Layout
//                 }
//             }

//             viscous_power_density_curnier_gradient(  // Model parameters
//                 mu,
//                 lambda,
//                 // X-coordinates
//                 xyz[0][i0],
//                 xyz[0][i1],
//                 xyz[0][i2],
//                 xyz[0][i3],
//                 // Y-coordinates
//                 xyz[1][i0],
//                 xyz[1][i1],
//                 xyz[1][i2],
//                 xyz[1][i3],
//                 // Z-coordinates
//                 xyz[2][i0],
//                 xyz[2][i1],
//                 xyz[2][i2],
//                 xyz[2][i3],
//                 // element dispalcement
//                 element_displacement,
//                 // output matrix
//                 element_vector);

//             for (int edof_i = 0; edof_i < 4; ++edof_i) {
//                 const idx_t dof = elems[edof_i][i] * block_size;

//                 for (int b = 0; b < block_size; b++) {
//                     // values[dof + b] += element_vector[b * 4 + edof_i];
// #pragma omp atomic update
//                     values[dof + b] += element_vector[edof_i * block_size + b];
//                 }
//             }
//         }
//     }

//     double tock = MPI_Wtime();
//     printf(
//         "viscous_power_density_curnier.c: viscous_power_density_curnier_assemble_gradient\t%g "
//         "seconds\n",
//         tock - tick);
// }

// void viscous_power_density_curnier_assemble_value(const ptrdiff_t nelements,
//                                                   const ptrdiff_t nnodes,
//                                                   idx_t *const SFEM_RESTRICT elems[4],
//                                                   geom_t *const SFEM_RESTRICT xyz[3],
//                                                   const real_t mu,
//                                                   const real_t lambda,
//                                                   const real_t eta,
//                                                   const real_t dt,
//                                                   const real_t *const SFEM_RESTRICT
//                                                       displacement_old,
//                                                   const real_t *const SFEM_RESTRICT displacement,
//                                                   real_t *const SFEM_RESTRICT value) {
//     SFEM_UNUSED(nnodes);

//     double tick = MPI_Wtime();
//     static const int block_size = 3;

// #pragma omp parallel
//     {
// #pragma omp for  // nowait
//         for (ptrdiff_t i = 0; i < nelements; ++i) {
//             idx_t ev[4];
//             idx_t ks[4];

//             real_t element_displacement[(4 * 3)];

// #pragma unroll(4)
//             for (int v = 0; v < 4; ++v) {
//                 ev[v] = elems[v][i];
//             }

//             // Element indices
//             const idx_t i0 = ev[0];
//             const idx_t i1 = ev[1];
//             const idx_t i2 = ev[2];
//             const idx_t i3 = ev[3];

//             for (int edof_i = 0; edof_i < 4; ++edof_i) {
//                 idx_t dof = ev[edof_i] * block_size;

//                 for (int b = 0; b < block_size; ++b) {
//                     // element_displacement[b * 4 + edof_i] = displacement[dof + b];
//                     element_displacement[b + edof_i * block_size] =
//                         displacement[dof + b];  // OLD
//                                                 // Layout
//                 }
//             }

//             real_t element_scalar = 0;
//             viscous_power_density_curnier_value(  // Model parameters
//                 mu,
//                 lambda,
//                 // X-coordinates
//                 xyz[0][i0],
//                 xyz[0][i1],
//                 xyz[0][i2],
//                 xyz[0][i3],
//                 // Y-coordinates
//                 xyz[1][i0],
//                 xyz[1][i1],
//                 xyz[1][i2],
//                 xyz[1][i3],
//                 // Z-coordinates
//                 xyz[2][i0],
//                 xyz[2][i1],
//                 xyz[2][i2],
//                 xyz[2][i3],
//                 // element dispalcement
//                 element_displacement,
//                 // output matrix
//                 &element_scalar);
// #pragma omp atomic update
//             (*value) += element_scalar;
//         }
//     }

//     double tock = MPI_Wtime();
//     printf(
//         "viscous_power_density_curnier.c: viscous_power_density_curnier_assemble_value\t%g "
//         "seconds\n",
//         tock - tick);
// }
