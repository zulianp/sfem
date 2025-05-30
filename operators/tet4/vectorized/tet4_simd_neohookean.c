#include "tet4_neohookean.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// static SFEM_INLINE void neohookean_energy(const real_t mu,
//                                           const real_t lambda,
//                                           const real_t px0,
//                                           const real_t px1,
//                                           const real_t px2,
//                                           const real_t px3,
//                                           const real_t py0,
//                                           const real_t py1,
//                                           const real_t py2,
//                                           const real_t py3,
//                                           const real_t pz0,
//                                           const real_t pz1,
//                                           const real_t pz2,
//                                           const real_t pz3,
//                                           const real_t *u,
//                                           real_t *element_energy) {
//     // FLOATING POINT OPS!
//     //	- Result: 3*ADD + ASSIGNMENT + 10*MUL + 10*POW
//     //	- Subexpressions: 38*ADD + 4*DIV + LOG + 84*MUL + 13*NEG + 29*SUB
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
//     const real_t x57 = u[11] * x26;
//     const real_t x58 = u[5] * x26;
//     const real_t x59 = u[8] * x26;
//     const real_t x60 = u[2] * x26;
//     const real_t x61 = x36 * x58 + x39 * x59 + x40 * x57 + x41 * x60;
//     const real_t x62 = x49 * x57 + x51 * x58 + x54 * x59 + x55 * x60;
//     const real_t x63 = x27 * x51 + x29 * x54 + x32 * x49 + x34 * x55 + 1;
//     const real_t x64 = x36 * x44 + x39 * x45 + x40 * x43 + x41 * x46 + 1;
//     const real_t x65 = x21 * x58 + x28 * x59 + x31 * x57 + x33 * x60 + 1;
//     const real_t x66 =
//         vlog(x35 * x56 * x61 - x35 * x62 * x64 + x42 * x47 * x62 - x42 * x56 * x65 - x47 * x61 * x63 + x63 * x64 *
//         x65);
//     *element_energy =
//         ((1.0 / 2.0) * lambda * pow(x66, 2) - mu * x66 +
//          (1.0 / 2.0) * mu *
//              (pow(x35, 2) + pow(x42, 2) + pow(x47, 2) + pow(x56, 2) + pow(x61, 2) + pow(x62, 2) + pow(x63, 2) +
//               pow(x64, 2) + pow(x65, 2) - 3)) *
//         (-x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 - x15 * x17 * x3 - x5 * x7 * x9);
// }

// static SFEM_INLINE void neohookean_gradient(const real_t mu,
//                                             const real_t lambda,
//                                             const real_t px0,
//                                             const real_t px1,
//                                             const real_t px2,
//                                             const real_t px3,
//                                             const real_t py0,
//                                             const real_t py1,
//                                             const real_t py2,
//                                             const real_t py3,
//                                             const real_t pz0,
//                                             const real_t pz1,
//                                             const real_t pz2,
//                                             const real_t pz3,
//                                             const real_t *u,
//                                             real_t *element_vector) {
//     // FLOATING POINT OPS!
//     //	- Result: 12*ADD + 12*ASSIGNMENT + 48*MUL
//     //	- Subexpressions: 49*ADD + 5*DIV + LOG + 151*MUL + 13*NEG + 50*SUB
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
//     const real_t x18 = -x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 - x15 * x17 * x3 - x5 * x7 *
//     x9; const real_t x19 = x0 * x12 - x16 * x8; const real_t x20 = x2 * x4; const real_t x21 = x12 * x6; const real_t
//     x22 = x10 * x16; const real_t x23 = x4 * x6; const real_t x24 = x10 * x12; const real_t x25 = x16 * x2; const
//     real_t x26 = 1.0 / (x0 * x20 - x0 * x24 + x14 * x21 - x14 * x25 + x22 * x8 - x23 * x8); const real_t x27 = u[3] *
//     x26; const real_t x28 = x0 * x4; const real_t x29 = x14 * x16; const real_t x30 = -x28 + x29; const real_t x31 =
//     u[6] * x26; const real_t x32 = -x12 * x14 + x4 * x8; const real_t x33 = u[9] * x26; const real_t x34 = -x19 + x28
//     - x29 - x32; const real_t x35 = u[0] * x26; const real_t x36 = x19 * x27 + x30 * x31 + x32 * x33 + x34 * x35;
//     const real_t x37 = x20 - x24;
//     const real_t x38 = -x37;
//     const real_t x39 = u[10] * x26;
//     const real_t x40 = x21 - x25;
//     const real_t x41 = -x40;
//     const real_t x42 = u[4] * x26;
//     const real_t x43 = -x22 + x23;
//     const real_t x44 = u[7] * x26;
//     const real_t x45 = x22 - x23 + x37 + x40;
//     const real_t x46 = u[1] * x26;
//     const real_t x47 = x38 * x39 + x41 * x42 + x43 * x44 + x45 * x46;
//     const real_t x48 = x10 * x8 - x14 * x2;
//     const real_t x49 = -x48;
//     const real_t x50 = u[11] * x26;
//     const real_t x51 = x0 * x2 - x6 * x8;
//     const real_t x52 = -x51;
//     const real_t x53 = u[5] * x26;
//     const real_t x54 = x0 * x10;
//     const real_t x55 = x14 * x6;
//     const real_t x56 = x54 - x55;
//     const real_t x57 = u[8] * x26;
//     const real_t x58 = x48 + x51 - x54 + x55;
//     const real_t x59 = u[2] * x26;
//     const real_t x60 = x49 * x50 + x52 * x53 + x56 * x57 + x58 * x59;
//     const real_t x61 = x47 * x60;
//     const real_t x62 = x39 * x49 + x42 * x52 + x44 * x56 + x46 * x58;
//     const real_t x63 = x38 * x50 + x41 * x53 + x43 * x57 + x45 * x59 + 1;
//     const real_t x64 = x62 * x63;
//     const real_t x65 = x61 - x64;
//     const real_t x66 = x27 * x41 + x31 * x43 + x33 * x38 + x35 * x45;
//     const real_t x67 = x19 * x53 + x30 * x57 + x32 * x50 + x34 * x59;
//     const real_t x68 = x62 * x67;
//     const real_t x69 = x19 * x42 + x30 * x44 + x32 * x39 + x34 * x46 + 1;
//     const real_t x70 = x60 * x69;
//     const real_t x71 = x27 * x52 + x31 * x56 + x33 * x49 + x35 * x58 + 1;
//     const real_t x72 = x47 * x67;
//     const real_t x73 = x36 * x61 - x36 * x64 + x63 * x69 * x71 + x66 * x68 - x66 * x70 - x71 * x72;
//     const real_t x74 = 1.0 / x73;
//     const real_t x75 = mu * x74;
//     const real_t x76 = lambda * x74 * vlog(x73);
//     const real_t x77 = mu * x36 - x65 * x75 + x65 * x76;
//     const real_t x78 = x26 * x34;
//     const real_t x79 = x68 - x70;
//     const real_t x80 = mu * x66 - x75 * x79 + x76 * x79;
//     const real_t x81 = x26 * x45;
//     const real_t x82 = x63 * x69 - x72;
//     const real_t x83 = mu * x71 - x75 * x82 + x76 * x82;
//     const real_t x84 = x26 * x58;
//     const real_t x85 = -x36 * x63 + x66 * x67;
//     const real_t x86 = mu * x62 - x75 * x85 + x76 * x85;
//     const real_t x87 = x36 * x60 - x67 * x71;
//     const real_t x88 = mu * x47 - x75 * x87 + x76 * x87;
//     const real_t x89 = -x60 * x66 + x63 * x71;
//     const real_t x90 = mu * x69 - x75 * x89 + x76 * x89;
//     const real_t x91 = -x47 * x71 + x62 * x66;
//     const real_t x92 = mu * x67 - x75 * x91 + x76 * x91;
//     const real_t x93 = x36 * x47 - x66 * x69;
//     const real_t x94 = mu * x60 - x75 * x93 + x76 * x93;
//     const real_t x95 = -x36 * x62 + x69 * x71;
//     const real_t x96 = mu * x63 - x75 * x95 + x76 * x95;
//     const real_t x97 = x26 * x41;
//     const real_t x98 = x19 * x26;
//     const real_t x99 = x26 * x52;
//     const real_t x100 = x26 * x43;
//     const real_t x101 = x26 * x30;
//     const real_t x102 = x26 * x56;
//     const real_t x103 = x26 * x38;
//     const real_t x104 = x26 * x32;
//     const real_t x105 = x26 * x49;
//     element_vector[0] = x18 * (x77 * x78 + x80 * x81 + x83 * x84);
//     element_vector[1] = x18 * (x78 * x90 + x81 * x88 + x84 * x86);
//     element_vector[2] = x18 * (x78 * x92 + x81 * x96 + x84 * x94);
//     element_vector[3] = x18 * (x77 * x98 + x80 * x97 + x83 * x99);
//     element_vector[4] = x18 * (x86 * x99 + x88 * x97 + x90 * x98);
//     element_vector[5] = x18 * (x92 * x98 + x94 * x99 + x96 * x97);
//     element_vector[6] = x18 * (x100 * x80 + x101 * x77 + x102 * x83);
//     element_vector[7] = x18 * (x100 * x88 + x101 * x90 + x102 * x86);
//     element_vector[8] = x18 * (x100 * x96 + x101 * x92 + x102 * x94);
//     element_vector[9] = x18 * (x103 * x80 + x104 * x77 + x105 * x83);
//     element_vector[10] = x18 * (x103 * x88 + x104 * x90 + x105 * x86);
//     element_vector[11] = x18 * (x103 * x96 + x104 * x92 + x105 * x94);
// }

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

static SFEM_INLINE void find_cols4(const idx_t *targets, const idx_t *const row, const int lenrow, idx_t *ks) {
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

static SFEM_INLINE vec_t vpow2(const vec_t x) { return x * x; }

static SFEM_INLINE vec_t rvpow2(const vec_t x) { return 1. / (x * x); }

static SFEM_INLINE vec_t vlog(const vec_t x) {
    vec_t ret;
#pragma unroll(SFEM_VEC_SIZE)
    for (int vi = 0; vi < SFEM_VEC_SIZE; ++vi) {
        const double xi = x[vi];
        ret[vi] = log(xi);
    }

    return ret;
}

static SFEM_INLINE void neohookean_hessian(const vec_t mu,
                                           const vec_t lambda,
                                           const vec_t px0,
                                           const vec_t px1,
                                           const vec_t px2,
                                           const vec_t px3,
                                           const vec_t py0,
                                           const vec_t py1,
                                           const vec_t py2,
                                           const vec_t py3,
                                           const vec_t pz0,
                                           const vec_t pz1,
                                           const vec_t pz2,
                                           const vec_t pz3,
                                           const vec_t *const  SFEM_RESTRICT u,
                                           vec_t * const SFEM_RESTRICT element_matrix) {
    // FLOATING POINT OPS!
    //	- Result: 21*ADD + 144*ASSIGNMENT + 75*MUL
    //	- Subexpressions: 460*ADD + 5*DIV + LOG + 914*MUL + 23*NEG + 10*POW + 140*SUB
    const vec_t x0 = pz0 - pz3;
    const vec_t x1 = -x0;
    const vec_t x2 = py0 - py2;
    const vec_t x3 = -x2;
    const vec_t x4 = px0 - px1;
    const vec_t x5 = (scalar_t)(-1.0 / 6.0) * x4;
    const vec_t x6 = py0 - py3;
    const vec_t x7 = -x6;
    const vec_t x8 = pz0 - pz2;
    const vec_t x9 = -x8;
    const vec_t x10 = py0 - py1;
    const vec_t x11 = -x10;
    const vec_t x12 = px0 - px2;
    const vec_t x13 = (scalar_t)(-1.0 / 6.0) * x12;
    const vec_t x14 = pz0 - pz1;
    const vec_t x15 = -x14;
    const vec_t x16 = px0 - px3;
    const vec_t x17 = (scalar_t)(-1.0 / 6.0) * x16;
    const vec_t x18 =
        -x1 * x11 * x13 + x1 * x3 * x5 + x11 * x17 * x9 + x13 * x15 * x7 - x15 * x17 * x3 - x5 * x7 * x9;
    const vec_t x19 = x2 * x4;
    const vec_t x20 = x10 * x12;
    const vec_t x21 = x19 - x20;
    const vec_t x22 = -x21;
    const vec_t x23 = x12 * x6;
    const vec_t x24 = x10 * x16;
    const vec_t x25 = x4 * x6;
    const vec_t x26 = x16 * x2;
    const vec_t x27 = (scalar_t)(1.0) / (x0 * x19 - x0 * x20 + x14 * x23 - x14 * x26 + x24 * x8 - x25 * x8);
    const vec_t x28 = u[10] * x27;
    const vec_t x29 = x23 - x26;
    const vec_t x30 = -x29;
    const vec_t x31 = u[4] * x27;
    const vec_t x32 = -x24 + x25;
    const vec_t x33 = u[7] * x27;
    const vec_t x34 = x21 + x24 - x25 + x29;
    const vec_t x35 = u[1] * x27;
    const vec_t x36 = x22 * x28 + x30 * x31 + x32 * x33 + x34 * x35;
    const vec_t x37 = x10 * x8 - x14 * x2;
    const vec_t x38 = -x37;
    const vec_t x39 = u[11] * x27;
    const vec_t x40 = x0 * x2 - x6 * x8;
    const vec_t x41 = -x40;
    const vec_t x42 = u[5] * x27;
    const vec_t x43 = x0 * x10;
    const vec_t x44 = x14 * x6;
    const vec_t x45 = x43 - x44;
    const vec_t x46 = u[8] * x27;
    const vec_t x47 = x37 + x40 - x43 + x44;
    const vec_t x48 = u[2] * x27;
    const vec_t x49 = x38 * x39 + x41 * x42 + x45 * x46 + x47 * x48;
    const vec_t x50 = x36 * x49;
    const vec_t x51 = x28 * x38 + x31 * x41 + x33 * x45 + x35 * x47;
    const vec_t x52 = x22 * x39 + x30 * x42 + x32 * x46 + x34 * x48 + 1;
    const vec_t x53 = x51 * x52;
    const vec_t x54 = x50 - x53;
    const vec_t x55 = u[3] * x27;
    const vec_t x56 = u[6] * x27;
    const vec_t x57 = u[9] * x27;
    const vec_t x58 = u[0] * x27;
    const vec_t x59 = x22 * x57 + x30 * x55 + x32 * x56 + x34 * x58;
    const vec_t x60 = -x12 * x14 + x4 * x8;
    const vec_t x61 = x0 * x12 - x16 * x8;
    const vec_t x62 = x0 * x4;
    const vec_t x63 = x14 * x16;
    const vec_t x64 = -x62 + x63;
    const vec_t x65 = -x60 - x61 + x62 - x63;
    const vec_t x66 = x39 * x60 + x42 * x61 + x46 * x64 + x48 * x65;
    const vec_t x67 = x51 * x66;
    const vec_t x68 = x55 * x61 + x56 * x64 + x57 * x60 + x58 * x65;
    const vec_t x69 = x28 * x60 + x31 * x61 + x33 * x64 + x35 * x65 + 1;
    const vec_t x70 = x49 * x69;
    const vec_t x71 = x38 * x57 + x41 * x55 + x45 * x56 + x47 * x58 + 1;
    const vec_t x72 = x36 * x66;
    const vec_t x73 = x50 * x68 + x52 * x69 * x71 - x53 * x68 + x59 * x67 - x59 * x70 - x71 * x72;
    const vec_t x74 = rvpow2(x73);
    const vec_t x75 = lambda * x74;
    const vec_t x76 = -x54;
    const vec_t x77 = mu * x74;
    const vec_t x78 = x76 * x77;
    const vec_t x79 = x54 * x75;
    const vec_t x80 = vlog(x73);
    const vec_t x81 = x76 * x80;
    const vec_t x82 = mu + vpow2(x54) * x75 - x54 * x78 + x79 * x81;
    const vec_t x83 = x27 * x65;
    const vec_t x84 = x67 - x70;
    const vec_t x85 = x75 * x84;
    const vec_t x86 = x54 * x85;
    const vec_t x87 = -x78 * x84 + x81 * x85 + x86;
    const vec_t x88 = x27 * x34;
    const vec_t x89 = -x52 * x69 + x72;
    const vec_t x90 = -x89;
    const vec_t x91 = x75 * x90;
    const vec_t x92 = x54 * x91;
    const vec_t x93 = -x78 * x90 + x81 * x91 + x92;
    const vec_t x94 = x27 * x47;
    const vec_t x95 = x82 * x83 + x87 * x88 + x93 * x94;
    const vec_t x96 = -x84;
    const vec_t x97 = x77 * x96;
    const vec_t x98 = x80 * x96;
    const vec_t x99 = mu + x75 * vpow2(x84) - x84 * x97 + x85 * x98;
    const vec_t x100 = x54 * x96;
    const vec_t x101 = x75 * x80;
    const vec_t x102 = x100 * x101 - x100 * x77 + x86;
    const vec_t x103 = x85 * x90;
    const vec_t x104 = x103 - x90 * x97 + x91 * x98;
    const vec_t x105 = x102 * x83 + x104 * x94 + x88 * x99;
    const vec_t x106 = x77 * x89;
    const vec_t x107 = x80 * x89;
    const vec_t x108 = mu - x106 * x90 + x107 * x91 + x75 * vpow2(x90);
    const vec_t x109 = -x106 * x54 + x107 * x79 + x92;
    const vec_t x110 = x103 - x106 * x84 + x107 * x85;
    const vec_t x111 = x108 * x94 + x109 * x83 + x110 * x88;
    const vec_t x112 = x49 * x68 - x66 * x71;
    const vec_t x113 = x112 * x85;
    const vec_t x114 = -x112;
    const vec_t x115 = x114 * x77;
    const vec_t x116 = x114 * x80;
    const vec_t x117 = x113 - x115 * x84 + x116 * x85;
    const vec_t x118 = (scalar_t)1.0 / x73;
    const vec_t x119 = mu * x118;
    const vec_t x120 = x119 * x49;
    const vec_t x121 = lambda * x118 * x80;
    const vec_t x122 = x121 * x49;
    const vec_t x123 = x112 * x79 - x120 + x122;
    const vec_t x124 = -x115 * x54 + x116 * x79 + x123;
    const vec_t x125 = x119 * x66;
    const vec_t x126 = x121 * x66;
    const vec_t x127 = x112 * x91 + x125 - x126;
    const vec_t x128 = -x115 * x90 + x116 * x91 + x127;
    const vec_t x129 = x117 * x88 + x124 * x83 + x128 * x94;
    const vec_t x130 = -x52 * x68 + x59 * x66;
    const vec_t x131 = x130 * x91;
    const vec_t x132 = -x130;
    const vec_t x133 = x132 * x77;
    const vec_t x134 = x132 * x80;
    const vec_t x135 = x131 - x133 * x90 + x134 * x91;
    const vec_t x136 = -x125 + x126 + x130 * x85;
    const vec_t x137 = -x133 * x84 + x134 * x85 + x136;
    const vec_t x138 = x119 * x52;
    const vec_t x139 = x121 * x52;
    const vec_t x140 = x130 * x79 + x138 - x139;
    const vec_t x141 = -x133 * x54 + x134 * x79 + x140;
    const vec_t x142 = x135 * x94 + x137 * x88 + x141 * x83;
    const vec_t x143 = x49 * x59 - x52 * x71;
    const vec_t x144 = -x143;
    const vec_t x145 = x144 * x79;
    const vec_t x146 = x143 * x77;
    const vec_t x147 = x143 * x80;
    const vec_t x148 = x145 - x146 * x54 + x147 * x79;
    const vec_t x149 = x120 - x122 + x144 * x85;
    const vec_t x150 = -x146 * x84 + x147 * x85 + x149;
    const vec_t x151 = -x138 + x139 + x144 * x91;
    const vec_t x152 = -x146 * x90 + x147 * x91 + x151;
    const vec_t x153 = x148 * x83 + x150 * x88 + x152 * x94;
    const vec_t x154 = x18 * (x129 * x88 + x142 * x94 + x153 * x83);
    const vec_t x155 = -x36 * x71 + x51 * x59;
    const vec_t x156 = x155 * x79;
    const vec_t x157 = -x155;
    const vec_t x158 = x157 * x77;
    const vec_t x159 = x157 * x80;
    const vec_t x160 = x156 - x158 * x54 + x159 * x79;
    const vec_t x161 = x119 * x51;
    const vec_t x162 = x121 * x51;
    const vec_t x163 = x155 * x85 - x161 + x162;
    const vec_t x164 = -x158 * x84 + x159 * x85 + x163;
    const vec_t x165 = x119 * x36;
    const vec_t x166 = x121 * x36;
    const vec_t x167 = x155 * x91 + x165 - x166;
    const vec_t x168 = -x158 * x90 + x159 * x91 + x167;
    const vec_t x169 = x160 * x83 + x164 * x88 + x168 * x94;
    const vec_t x170 = -x36 * x68 + x59 * x69;
    const vec_t x171 = -x170;
    const vec_t x172 = x171 * x91;
    const vec_t x173 = x170 * x77;
    const vec_t x174 = x170 * x80;
    const vec_t x175 = x172 - x173 * x90 + x174 * x91;
    const vec_t x176 = -x165 + x166 + x171 * x79;
    const vec_t x177 = -x173 * x54 + x174 * x79 + x176;
    const vec_t x178 = x119 * x69;
    const vec_t x179 = x121 * x69;
    const vec_t x180 = x171 * x85 + x178 - x179;
    const vec_t x181 = -x173 * x84 + x174 * x85 + x180;
    const vec_t x182 = x175 * x94 + x177 * x83 + x181 * x88;
    const vec_t x183 = x51 * x68 - x69 * x71;
    const vec_t x184 = -x183;
    const vec_t x185 = x184 * x85;
    const vec_t x186 = x183 * x77;
    const vec_t x187 = x183 * x80;
    const vec_t x188 = x185 - x186 * x84 + x187 * x85;
    const vec_t x189 = x161 - x162 + x184 * x79;
    const vec_t x190 = -x186 * x54 + x187 * x79 + x189;
    const vec_t x191 = -x178 + x179 + x184 * x91;
    const vec_t x192 = -x186 * x90 + x187 * x91 + x191;
    const vec_t x193 = x188 * x88 + x190 * x83 + x192 * x94;
    const vec_t x194 = x18 * (x169 * x83 + x182 * x94 + x193 * x88);
    const vec_t x195 = x27 * x30;
    const vec_t x196 = x27 * x61;
    const vec_t x197 = x27 * x41;
    const vec_t x198 = x18 * (x105 * x195 + x111 * x197 + x196 * x95);
    const vec_t x199 = x18 * (x129 * x195 + x142 * x197 + x153 * x196);
    const vec_t x200 = x18 * (x169 * x196 + x182 * x197 + x193 * x195);
    const vec_t x201 = x27 * x32;
    const vec_t x202 = x27 * x64;
    const vec_t x203 = x27 * x45;
    const vec_t x204 = x18 * (x105 * x201 + x111 * x203 + x202 * x95);
    const vec_t x205 = x18 * (x129 * x201 + x142 * x203 + x153 * x202);
    const vec_t x206 = x18 * (x169 * x202 + x182 * x203 + x193 * x201);
    const vec_t x207 = x22 * x27;
    const vec_t x208 = x27 * x60;
    const vec_t x209 = x27 * x38;
    const vec_t x210 = x18 * (x105 * x207 + x111 * x209 + x208 * x95);
    const vec_t x211 = x18 * (x129 * x207 + x142 * x209 + x153 * x208);
    const vec_t x212 = x18 * (x169 * x208 + x182 * x209 + x193 * x207);
    const vec_t x213 = x130 * x75;
    const vec_t x214 = mu + vpow2(x130) * x75 - x130 * x133 + x134 * x213;
    const vec_t x215 = x112 * x75;
    const vec_t x216 = x130 * x215;
    const vec_t x217 = -x112 * x133 + x134 * x215 + x216;
    const vec_t x218 = x144 * x75;
    const vec_t x219 = x130 * x218;
    const vec_t x220 = -x133 * x144 + x134 * x218 + x219;
    const vec_t x221 = x214 * x94 + x217 * x88 + x220 * x83;
    const vec_t x222 = mu + vpow2(x112) * x75 - x112 * x115 + x116 * x215;
    const vec_t x223 = -x115 * x130 + x116 * x213 + x216;
    const vec_t x224 = x144 * x215;
    const vec_t x225 = -x115 * x144 + x116 * x218 + x224;
    const vec_t x226 = x222 * x88 + x223 * x94 + x225 * x83;
    const vec_t x227 = mu + vpow2(x144) * x75 - x144 * x146 + x147 * x218;
    const vec_t x228 = -x130 * x146 + x147 * x213 + x219;
    const vec_t x229 = -x112 * x146 + x147 * x215 + x224;
    const vec_t x230 = x227 * x83 + x228 * x94 + x229 * x88;
    const vec_t x231 = x171 * x213;
    const vec_t x232 = -x130 * x173 + x174 * x213 + x231;
    const vec_t x233 = x119 * x68;
    const vec_t x234 = x121 * x68;
    const vec_t x235 = x171 * x215 - x233 + x234;
    const vec_t x236 = -x112 * x173 + x174 * x215 + x235;
    const vec_t x237 = x119 * x59;
    const vec_t x238 = x121 * x59;
    const vec_t x239 = x171 * x218 + x237 - x238;
    const vec_t x240 = -x144 * x173 + x174 * x218 + x239;
    const vec_t x241 = x232 * x94 + x236 * x88 + x240 * x83;
    const vec_t x242 = x155 * x218;
    const vec_t x243 = -x144 * x158 + x159 * x218 + x242;
    const vec_t x244 = x155 * x213 - x237 + x238;
    const vec_t x245 = -x130 * x158 + x159 * x213 + x244;
    const vec_t x246 = x119 * x71;
    const vec_t x247 = x121 * x71;
    const vec_t x248 = x155 * x215 + x246 - x247;
    const vec_t x249 = -x112 * x158 + x159 * x215 + x248;
    const vec_t x250 = x243 * x83 + x245 * x94 + x249 * x88;
    const vec_t x251 = x184 * x215;
    const vec_t x252 = -x112 * x186 + x187 * x215 + x251;
    const vec_t x253 = x184 * x213 + x233 - x234;
    const vec_t x254 = -x130 * x186 + x187 * x213 + x253;
    const vec_t x255 = x184 * x218 - x246 + x247;
    const vec_t x256 = -x144 * x186 + x187 * x218 + x255;
    const vec_t x257 = x252 * x88 + x254 * x94 + x256 * x83;
    const vec_t x258 = x18 * (x241 * x94 + x250 * x83 + x257 * x88);
    const vec_t x259 = -x112 * x97 + x113 + x215 * x98;
    const vec_t x260 = -x130 * x97 + x136 + x213 * x98;
    const vec_t x261 = -x144 * x97 + x149 + x218 * x98;
    const vec_t x262 = x259 * x88 + x260 * x94 + x261 * x83;
    const vec_t x263 = -x144 * x78 + x145 + x218 * x81;
    const vec_t x264 = -x112 * x78 + x123 + x215 * x81;
    const vec_t x265 = -x130 * x78 + x140 + x213 * x81;
    const vec_t x266 = x263 * x83 + x264 * x88 + x265 * x94;
    const vec_t x267 = -x106 * x130 + x107 * x213 + x131;
    const vec_t x268 = -x106 * x112 + x107 * x215 + x127;
    const vec_t x269 = -x106 * x144 + x107 * x218 + x151;
    const vec_t x270 = x267 * x94 + x268 * x88 + x269 * x83;
    const vec_t x271 = x18 * (x195 * x262 + x196 * x266 + x197 * x270);
    const vec_t x272 = x18 * (x195 * x226 + x196 * x230 + x197 * x221);
    const vec_t x273 = x18 * (x195 * x257 + x196 * x250 + x197 * x241);
    const vec_t x274 = x18 * (x201 * x262 + x202 * x266 + x203 * x270);
    const vec_t x275 = x18 * (x201 * x226 + x202 * x230 + x203 * x221);
    const vec_t x276 = x18 * (x201 * x257 + x202 * x250 + x203 * x241);
    const vec_t x277 = x18 * (x207 * x262 + x208 * x266 + x209 * x270);
    const vec_t x278 = x18 * (x207 * x226 + x208 * x230 + x209 * x221);
    const vec_t x279 = x18 * (x207 * x257 + x208 * x250 + x209 * x241);
    const vec_t x280 = x171 * x75;
    const vec_t x281 = mu + vpow2(x171) * x75 - x171 * x173 + x174 * x280;
    const vec_t x282 = x155 * x280;
    const vec_t x283 = x155 * x75;
    const vec_t x284 = -x155 * x173 + x174 * x283 + x282;
    const vec_t x285 = x184 * x280;
    const vec_t x286 = x101 * x184;
    const vec_t x287 = x170 * x286 - x173 * x184 + x285;
    const vec_t x288 = x281 * x94 + x284 * x83 + x287 * x88;
    const vec_t x289 = mu + vpow2(x155) * x75 - x155 * x158 + x159 * x283;
    const vec_t x290 = -x158 * x171 + x159 * x280 + x282;
    const vec_t x291 = x184 * x283;
    const vec_t x292 = x157 * x286 - x158 * x184 + x291;
    const vec_t x293 = x289 * x83 + x290 * x94 + x292 * x88;
    const vec_t x294 = mu + x183 * x286 + vpow2(x184) * x75 - x184 * x186;
    const vec_t x295 = -x155 * x186 + x187 * x283 + x291;
    const vec_t x296 = -x171 * x186 + x187 * x280 + x285;
    const vec_t x297 = x294 * x88 + x295 * x83 + x296 * x94;
    const vec_t x298 = -x155 * x78 + x156 + x283 * x81;
    const vec_t x299 = -x171 * x78 + x176 + x280 * x81;
    const vec_t x300 = -x184 * x78 + x189 + x286 * x76;
    const vec_t x301 = x298 * x83 + x299 * x94 + x300 * x88;
    const vec_t x302 = -x184 * x97 + x185 + x286 * x96;
    const vec_t x303 = -x155 * x97 + x163 + x283 * x98;
    const vec_t x304 = -x171 * x97 + x180 + x280 * x98;
    const vec_t x305 = x302 * x88 + x303 * x83 + x304 * x94;
    const vec_t x306 = -x106 * x171 + x107 * x280 + x172;
    const vec_t x307 = -x106 * x155 + x107 * x283 + x167;
    const vec_t x308 = -x106 * x184 + x191 + x286 * x89;
    const vec_t x309 = x306 * x94 + x307 * x83 + x308 * x88;
    const vec_t x310 = x18 * (x195 * x305 + x196 * x301 + x197 * x309);
    const vec_t x311 = -x133 * x171 + x134 * x280 + x231;
    const vec_t x312 = -x133 * x155 + x134 * x283 + x244;
    const vec_t x313 = x132 * x286 - x133 * x184 + x253;
    const vec_t x314 = x311 * x94 + x312 * x83 + x313 * x88;
    const vec_t x315 = x114 * x286 - x115 * x184 + x251;
    const vec_t x316 = -x115 * x171 + x116 * x280 + x235;
    const vec_t x317 = -x115 * x155 + x116 * x283 + x248;
    const vec_t x318 = x315 * x88 + x316 * x94 + x317 * x83;
    const vec_t x319 = -x146 * x155 + x147 * x283 + x242;
    const vec_t x320 = -x146 * x171 + x147 * x280 + x239;
    const vec_t x321 = x143 * x286 - x146 * x184 + x255;
    const vec_t x322 = x319 * x83 + x320 * x94 + x321 * x88;
    const vec_t x323 = x18 * (x195 * x318 + x196 * x322 + x197 * x314);
    const vec_t x324 = x18 * (x195 * x297 + x196 * x293 + x197 * x288);
    const vec_t x325 = x18 * (x201 * x305 + x202 * x301 + x203 * x309);
    const vec_t x326 = x18 * (x201 * x318 + x202 * x322 + x203 * x314);
    const vec_t x327 = x18 * (x201 * x297 + x202 * x293 + x203 * x288);
    const vec_t x328 = x18 * (x207 * x305 + x208 * x301 + x209 * x309);
    const vec_t x329 = x18 * (x207 * x318 + x208 * x322 + x209 * x314);
    const vec_t x330 = x18 * (x207 * x297 + x208 * x293 + x209 * x288);
    const vec_t x331 = x102 * x196 + x104 * x197 + x195 * x99;
    const vec_t x332 = x195 * x87 + x196 * x82 + x197 * x93;
    const vec_t x333 = x108 * x197 + x109 * x196 + x110 * x195;
    const vec_t x334 = x117 * x195 + x124 * x196 + x128 * x197;
    const vec_t x335 = x135 * x197 + x137 * x195 + x141 * x196;
    const vec_t x336 = x148 * x196 + x150 * x195 + x152 * x197;
    const vec_t x337 = x18 * (x195 * x334 + x196 * x336 + x197 * x335);
    const vec_t x338 = x160 * x196 + x164 * x195 + x168 * x197;
    const vec_t x339 = x175 * x197 + x177 * x196 + x181 * x195;
    const vec_t x340 = x188 * x195 + x190 * x196 + x192 * x197;
    const vec_t x341 = x18 * (x195 * x340 + x196 * x338 + x197 * x339);
    const vec_t x342 = x18 * (x201 * x331 + x202 * x332 + x203 * x333);
    const vec_t x343 = x18 * (x201 * x334 + x202 * x336 + x203 * x335);
    const vec_t x344 = x18 * (x201 * x340 + x202 * x338 + x203 * x339);
    const vec_t x345 = x18 * (x207 * x331 + x208 * x332 + x209 * x333);
    const vec_t x346 = x18 * (x207 * x334 + x208 * x336 + x209 * x335);
    const vec_t x347 = x18 * (x207 * x340 + x208 * x338 + x209 * x339);
    const vec_t x348 = x195 * x222 + x196 * x225 + x197 * x223;
    const vec_t x349 = x195 * x217 + x196 * x220 + x197 * x214;
    const vec_t x350 = x195 * x229 + x196 * x227 + x197 * x228;
    const vec_t x351 = x195 * x236 + x196 * x240 + x197 * x232;
    const vec_t x352 = x195 * x249 + x196 * x243 + x197 * x245;
    const vec_t x353 = x195 * x252 + x196 * x256 + x197 * x254;
    const vec_t x354 = x18 * (x195 * x353 + x196 * x352 + x197 * x351);
    const vec_t x355 = x195 * x259 + x196 * x261 + x197 * x260;
    const vec_t x356 = x195 * x264 + x196 * x263 + x197 * x265;
    const vec_t x357 = x195 * x268 + x196 * x269 + x197 * x267;
    const vec_t x358 = x18 * (x201 * x355 + x202 * x356 + x203 * x357);
    const vec_t x359 = x18 * (x201 * x348 + x202 * x350 + x203 * x349);
    const vec_t x360 = x18 * (x201 * x353 + x202 * x352 + x203 * x351);
    const vec_t x361 = x18 * (x207 * x355 + x208 * x356 + x209 * x357);
    const vec_t x362 = x18 * (x207 * x348 + x208 * x350 + x209 * x349);
    const vec_t x363 = x18 * (x207 * x353 + x208 * x352 + x209 * x351);
    const vec_t x364 = x195 * x292 + x196 * x289 + x197 * x290;
    const vec_t x365 = x195 * x287 + x196 * x284 + x197 * x281;
    const vec_t x366 = x195 * x294 + x196 * x295 + x197 * x296;
    const vec_t x367 = x195 * x300 + x196 * x298 + x197 * x299;
    const vec_t x368 = x195 * x302 + x196 * x303 + x197 * x304;
    const vec_t x369 = x195 * x308 + x196 * x307 + x197 * x306;
    const vec_t x370 = x18 * (x201 * x368 + x202 * x367 + x203 * x369);
    const vec_t x371 = x195 * x313 + x196 * x312 + x197 * x311;
    const vec_t x372 = x195 * x315 + x196 * x317 + x197 * x316;
    const vec_t x373 = x195 * x321 + x196 * x319 + x197 * x320;
    const vec_t x374 = x18 * (x201 * x372 + x202 * x373 + x203 * x371);
    const vec_t x375 = x18 * (x201 * x366 + x202 * x364 + x203 * x365);
    const vec_t x376 = x18 * (x207 * x368 + x208 * x367 + x209 * x369);
    const vec_t x377 = x18 * (x207 * x372 + x208 * x373 + x209 * x371);
    const vec_t x378 = x18 * (x207 * x366 + x208 * x364 + x209 * x365);
    const vec_t x379 = x102 * x202 + x104 * x203 + x201 * x99;
    const vec_t x380 = x201 * x87 + x202 * x82 + x203 * x93;
    const vec_t x381 = x108 * x203 + x109 * x202 + x110 * x201;
    const vec_t x382 = x117 * x201 + x124 * x202 + x128 * x203;
    const vec_t x383 = x135 * x203 + x137 * x201 + x141 * x202;
    const vec_t x384 = x148 * x202 + x150 * x201 + x152 * x203;
    const vec_t x385 = x18 * (x201 * x382 + x202 * x384 + x203 * x383);
    const vec_t x386 = x160 * x202 + x164 * x201 + x168 * x203;
    const vec_t x387 = x175 * x203 + x177 * x202 + x181 * x201;
    const vec_t x388 = x188 * x201 + x190 * x202 + x192 * x203;
    const vec_t x389 = x18 * (x201 * x388 + x202 * x386 + x203 * x387);
    const vec_t x390 = x18 * (x207 * x379 + x208 * x380 + x209 * x381);
    const vec_t x391 = x18 * (x207 * x382 + x208 * x384 + x209 * x383);
    const vec_t x392 = x18 * (x207 * x388 + x208 * x386 + x209 * x387);
    const vec_t x393 = x201 * x222 + x202 * x225 + x203 * x223;
    const vec_t x394 = x201 * x217 + x202 * x220 + x203 * x214;
    const vec_t x395 = x201 * x229 + x202 * x227 + x203 * x228;
    const vec_t x396 = x201 * x236 + x202 * x240 + x203 * x232;
    const vec_t x397 = x201 * x249 + x202 * x243 + x203 * x245;
    const vec_t x398 = x201 * x252 + x202 * x256 + x203 * x254;
    const vec_t x399 = x18 * (x201 * x398 + x202 * x397 + x203 * x396);
    const vec_t x400 =
        x18 * (x207 * (x201 * x259 + x202 * x261 + x203 * x260) + x208 * (x201 * x264 + x202 * x263 + x203 * x265) +
               x209 * (x201 * x268 + x202 * x269 + x203 * x267));
    const vec_t x401 = x18 * (x207 * x393 + x208 * x395 + x209 * x394);
    const vec_t x402 = x18 * (x207 * x398 + x208 * x397 + x209 * x396);
    const vec_t x403 = x201 * x292 + x202 * x289 + x203 * x290;
    const vec_t x404 = x201 * x287 + x202 * x284 + x203 * x281;
    const vec_t x405 = x201 * x294 + x202 * x295 + x203 * x296;
    const vec_t x406 =
        x18 * (x207 * (x201 * x302 + x202 * x303 + x203 * x304) + x208 * (x201 * x300 + x202 * x298 + x203 * x299) +
               x209 * (x201 * x308 + x202 * x307 + x203 * x306));
    const vec_t x407 =
        x18 * (x207 * (x201 * x315 + x202 * x317 + x203 * x316) + x208 * (x201 * x321 + x202 * x319 + x203 * x320) +
               x209 * (x201 * x313 + x202 * x312 + x203 * x311));
    const vec_t x408 = x18 * (x207 * x405 + x208 * x403 + x209 * x404);
    const vec_t x409 =
        x18 * (x207 * (x117 * x207 + x124 * x208 + x128 * x209) + x208 * (x148 * x208 + x150 * x207 + x152 * x209) +
               x209 * (x135 * x209 + x137 * x207 + x141 * x208));
    const vec_t x410 =
        x18 * (x207 * (x188 * x207 + x190 * x208 + x192 * x209) + x208 * (x160 * x208 + x164 * x207 + x168 * x209) +
               x209 * (x175 * x209 + x177 * x208 + x181 * x207));
    const vec_t x411 =
        x18 * (x207 * (x207 * x252 + x208 * x256 + x209 * x254) + x208 * (x207 * x249 + x208 * x243 + x209 * x245) +
               x209 * (x207 * x236 + x208 * x240 + x209 * x232));
    element_matrix[0] = x18 * (x105 * x88 + x111 * x94 + x83 * x95);
    element_matrix[1] = x154;
    element_matrix[2] = x194;
    element_matrix[3] = x198;
    element_matrix[4] = x199;
    element_matrix[5] = x200;
    element_matrix[6] = x204;
    element_matrix[7] = x205;
    element_matrix[8] = x206;
    element_matrix[9] = x210;
    element_matrix[10] = x211;
    element_matrix[11] = x212;
    element_matrix[12] = x154;
    element_matrix[13] = x18 * (x221 * x94 + x226 * x88 + x230 * x83);
    element_matrix[14] = x258;
    element_matrix[15] = x271;
    element_matrix[16] = x272;
    element_matrix[17] = x273;
    element_matrix[18] = x274;
    element_matrix[19] = x275;
    element_matrix[20] = x276;
    element_matrix[21] = x277;
    element_matrix[22] = x278;
    element_matrix[23] = x279;
    element_matrix[24] = x194;
    element_matrix[25] = x258;
    element_matrix[26] = x18 * (x288 * x94 + x293 * x83 + x297 * x88);
    element_matrix[27] = x310;
    element_matrix[28] = x323;
    element_matrix[29] = x324;
    element_matrix[30] = x325;
    element_matrix[31] = x326;
    element_matrix[32] = x327;
    element_matrix[33] = x328;
    element_matrix[34] = x329;
    element_matrix[35] = x330;
    element_matrix[36] = x198;
    element_matrix[37] = x271;
    element_matrix[38] = x310;
    element_matrix[39] = x18 * (x195 * x331 + x196 * x332 + x197 * x333);
    element_matrix[40] = x337;
    element_matrix[41] = x341;
    element_matrix[42] = x342;
    element_matrix[43] = x343;
    element_matrix[44] = x344;
    element_matrix[45] = x345;
    element_matrix[46] = x346;
    element_matrix[47] = x347;
    element_matrix[48] = x199;
    element_matrix[49] = x272;
    element_matrix[50] = x323;
    element_matrix[51] = x337;
    element_matrix[52] = x18 * (x195 * x348 + x196 * x350 + x197 * x349);
    element_matrix[53] = x354;
    element_matrix[54] = x358;
    element_matrix[55] = x359;
    element_matrix[56] = x360;
    element_matrix[57] = x361;
    element_matrix[58] = x362;
    element_matrix[59] = x363;
    element_matrix[60] = x200;
    element_matrix[61] = x273;
    element_matrix[62] = x324;
    element_matrix[63] = x341;
    element_matrix[64] = x354;
    element_matrix[65] = x18 * (x195 * x366 + x196 * x364 + x197 * x365);
    element_matrix[66] = x370;
    element_matrix[67] = x374;
    element_matrix[68] = x375;
    element_matrix[69] = x376;
    element_matrix[70] = x377;
    element_matrix[71] = x378;
    element_matrix[72] = x204;
    element_matrix[73] = x274;
    element_matrix[74] = x325;
    element_matrix[75] = x342;
    element_matrix[76] = x358;
    element_matrix[77] = x370;
    element_matrix[78] = x18 * (x201 * x379 + x202 * x380 + x203 * x381);
    element_matrix[79] = x385;
    element_matrix[80] = x389;
    element_matrix[81] = x390;
    element_matrix[82] = x391;
    element_matrix[83] = x392;
    element_matrix[84] = x205;
    element_matrix[85] = x275;
    element_matrix[86] = x326;
    element_matrix[87] = x343;
    element_matrix[88] = x359;
    element_matrix[89] = x374;
    element_matrix[90] = x385;
    element_matrix[91] = x18 * (x201 * x393 + x202 * x395 + x203 * x394);
    element_matrix[92] = x399;
    element_matrix[93] = x400;
    element_matrix[94] = x401;
    element_matrix[95] = x402;
    element_matrix[96] = x206;
    element_matrix[97] = x276;
    element_matrix[98] = x327;
    element_matrix[99] = x344;
    element_matrix[100] = x360;
    element_matrix[101] = x375;
    element_matrix[102] = x389;
    element_matrix[103] = x399;
    element_matrix[104] = x18 * (x201 * x405 + x202 * x403 + x203 * x404);
    element_matrix[105] = x406;
    element_matrix[106] = x407;
    element_matrix[107] = x408;
    element_matrix[108] = x210;
    element_matrix[109] = x277;
    element_matrix[110] = x328;
    element_matrix[111] = x345;
    element_matrix[112] = x361;
    element_matrix[113] = x376;
    element_matrix[114] = x390;
    element_matrix[115] = x400;
    element_matrix[116] = x406;
    element_matrix[117] =
        x18 * (x207 * (x102 * x208 + x104 * x209 + x207 * x99) + x208 * (x207 * x87 + x208 * x82 + x209 * x93) +
               x209 * (x108 * x209 + x109 * x208 + x110 * x207));
    element_matrix[118] = x409;
    element_matrix[119] = x410;
    element_matrix[120] = x211;
    element_matrix[121] = x278;
    element_matrix[122] = x329;
    element_matrix[123] = x346;
    element_matrix[124] = x362;
    element_matrix[125] = x377;
    element_matrix[126] = x391;
    element_matrix[127] = x401;
    element_matrix[128] = x407;
    element_matrix[129] = x409;
    element_matrix[130] =
        x18 * (x207 * (x207 * x222 + x208 * x225 + x209 * x223) + x208 * (x207 * x229 + x208 * x227 + x209 * x228) +
               x209 * (x207 * x217 + x208 * x220 + x209 * x214));
    element_matrix[131] = x411;
    element_matrix[132] = x212;
    element_matrix[133] = x279;
    element_matrix[134] = x330;
    element_matrix[135] = x347;
    element_matrix[136] = x363;
    element_matrix[137] = x378;
    element_matrix[138] = x392;
    element_matrix[139] = x402;
    element_matrix[140] = x408;
    element_matrix[141] = x410;
    element_matrix[142] = x411;
    element_matrix[143] =
        x18 * (x207 * (x207 * x294 + x208 * x295 + x209 * x296) + x208 * (x207 * x292 + x208 * x289 + x209 * x290) +
               x209 * (x207 * x287 + x208 * x284 + x209 * x281));
}

void neohookean_assemble_hessian(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t *const  SFEM_RESTRICT elems[4],
                                 geom_t *const  SFEM_RESTRICT xyz[3],
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t *const  SFEM_RESTRICT displacement,
                                 count_t *const  SFEM_RESTRICT rowptr,
                                 idx_t *const  SFEM_RESTRICT colidx,
                                 real_t *const  SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    vec_t x[4];
    vec_t y[4];
    vec_t z[4];

    idx_t ev[4];
    idx_t ks[4];

    vec_t element_matrix[(4 * 3) * (4 * 3)];
    vec_t element_displacement[(4 * 3)];

    vec_t vmu;
    vec_t vlambda;

    for (int vi = 0; vi < SFEM_VEC_SIZE; ++vi) {
        vmu[vi] = mu;
        vlambda[vi] = lambda;
    }

    static const int block_size = 3;
    static const int mat_block_size = 3 * 3;

    for (ptrdiff_t i = 0; i < nelements; i += SFEM_VEC_SIZE) {
        const int nvec = MAX(1, MIN(nelements - (i + SFEM_VEC_SIZE), SFEM_VEC_SIZE));

        for (int vi = 0; vi < nvec; ++vi) {
            const ptrdiff_t offset = i + vi;
            for (int d = 0; d < 4; ++d) {
                const idx_t vidx = elems[d][offset];
                x[d][vi] = xyz[0][vidx];
                y[d][vi] = xyz[1][vidx];
                z[d][vi] = xyz[2][vidx];

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[d * block_size + b][vi] = displacement[vidx * block_size + b];
                }
            }
        }

        neohookean_hessian(  // Model parameters
            vmu,
            vlambda,
            // X-coordinates
            x[0],
            x[1],
            x[2],
            x[3],
            // Y-coordinates
            y[0],
            y[1],
            y[2],
            y[3],
            // Z-coordinates
            z[0],
            z[1],
            z[2],
            z[3],
            // element dispalcement
            element_displacement,
            // output matrix
            element_matrix);

        for (int vi = 0; vi < nvec; ++vi) {
            const idx_t offset = i + vi;

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][offset];
            }

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][offset];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
                const idx_t *row = &colidx[rowptr[dof_i]];
                find_cols4(ev, row, lenrow, ks);

                // Blocks for row
                real_t *row_blocks = &values[rowptr[dof_i] * mat_block_size];

                for (int edof_j = 0; edof_j < 4; ++edof_j) {
                    // Block for column
                    const idx_t block_k = ks[edof_j] * mat_block_size;
                    real_t *block = &row_blocks[block_k];

                    // Iterate over dimensions
                    for (int bj = 0; bj < block_size; ++bj) {
                        const idx_t offset_j = bj * block_size;

                        for (int bi = 0; bi < block_size; ++bi) {
                            block[offset_j + bi] += element_matrix[(edof_i * block_size + bi) * block_size * 4 +
                                                                   edof_j * block_size + bj][vi];
                        }
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("simd_neohookean.c: neohookean_assemble_hessian\t%g seconds\n", tock - tick);
}
