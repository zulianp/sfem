#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

// Does not lead to improvements on Apple M1, however 2x on x86 with avx2
// #define SFEM_ENABLE_EXPLICIT_VECTORIZATION

#define POW2(l) ((l) * (l))
#define RPOW2(l) (1. / ((l) * (l)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static const int stride = 1;

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

static int check_symmetric(int n, const real_t *const element_matrix) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const real_t diff = element_matrix[i * n + j] - element_matrix[i + j * n];
            assert(diff < 1e-16);
            if (diff > 1e-16) {
                return 1;
            }

            // printf("%g ",  element_matrix[i*n + j] );
        }

        // printf("\n");
    }

    // printf("\n");

    return 0;
}

static SFEM_INLINE void tet4_linear_elasticity_assemble_value_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
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

    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT element_scalar) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = -px0 + px2;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz1;
    const real_t x7 = -px0 + px3;
    const real_t x8 = -py0 + py1;
    const real_t x9 = -pz0 + pz2;
    const real_t x10 = x8 * x9;
    const real_t x11 = x5 * x9;
    const real_t x12 = x2 * x8;
    const real_t x13 = x1 * x6;
    const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x7 - x12 * x4 - x13 * x7 + x4 * x5 * x6;
    const real_t x15 = 1.0 / x14;
    const real_t x16 = x15 * (-x11 + x3);
    const real_t x17 = x15 * (-x12 + x5 * x6);
    const real_t x18 = x15 * (x10 - x13);
    const real_t x19 = -x16 - x17 - x18;
    const real_t x20 = u[0] * x19 + u[1] * x16 + u[2] * x17 + u[3] * x18;
    const real_t x21 = POW2(x20);
    const real_t x22 = (1.0 / 12.0) * lambda;
    const real_t x23 = x15 * (-x2 * x4 + x7 * x9);
    const real_t x24 = x15 * (x0 * x2 - x6 * x7);
    const real_t x25 = x15 * (-x0 * x9 + x4 * x6);
    const real_t x26 = -x23 - x24 - x25;
    const real_t x27 = u[4] * x26 + u[5] * x23 + u[6] * x24 + u[7] * x25;
    const real_t x28 = POW2(x27);
    const real_t x29 = x15 * (-x0 * x5 + x7 * x8);
    const real_t x30 = x15 * (x0 * x1 - x4 * x8);
    const real_t x31 = x15 * (-x1 * x7 + x4 * x5);
    const real_t x32 = -x29 - x30 - x31;
    const real_t x33 = u[10] * x29 + u[11] * x30 + u[8] * x32 + u[9] * x31;
    const real_t x34 = POW2(x33);
    const real_t x35 = u[0] * x32 + u[1] * x31 + u[2] * x29 + u[3] * x30;
    const real_t x36 = (1.0 / 12.0) * mu;
    const real_t x37 = u[0] * x26 + u[1] * x23 + u[2] * x24 + u[3] * x25;
    const real_t x38 = (1.0 / 6.0) * mu;
    const real_t x39 = u[4] * x32 + u[5] * x31 + u[6] * x29 + u[7] * x30;
    const real_t x40 = u[4] * x19 + u[5] * x16 + u[6] * x17 + u[7] * x18;
    const real_t x41 = u[10] * x24 + u[11] * x25 + u[8] * x26 + u[9] * x23;
    const real_t x42 = u[10] * x17 + u[11] * x18 + u[8] * x19 + u[9] * x16;
    const real_t x43 = (1.0 / 6.0) * lambda * x20;
    element_scalar[0] =
        x14 * ((1.0 / 6.0) * lambda * x27 * x33 + x21 * x22 + x21 * x38 + x22 * x28 + x22 * x34 +
               x27 * x43 + x28 * x38 + x33 * x43 + x34 * x38 + POW2(x35) * x36 + x35 * x38 * x42 +
               x36 * POW2(x37) + x36 * POW2(x39) + x36 * POW2(x40) + x36 * POW2(x41) +
               x36 * POW2(x42) + x37 * x38 * x40 + x38 * x39 * x41);
}

static SFEM_INLINE void tet4_linear_elasticity_assemble_gradient_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
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
    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = -px0 + px2;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz1;
    const real_t x7 = -px0 + px3;
    const real_t x8 = -py0 + py1;
    const real_t x9 = -pz0 + pz2;
    const real_t x10 = x8 * x9;
    const real_t x11 = x5 * x9;
    const real_t x12 = x2 * x8;
    const real_t x13 = x1 * x6;
    const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x7 - x12 * x4 - x13 * x7 + x4 * x5 * x6;
    const real_t x15 = 1.0 / x14;
    const real_t x16 = x15 * (-x11 + x3);
    const real_t x17 = x15 * (-x12 + x5 * x6);
    const real_t x18 = x15 * (x10 - x13);
    const real_t x19 = -x16 - x17 - x18;
    const real_t x20 = u[0] * x19 + u[1] * x16 + u[2] * x17 + u[3] * x18;
    const real_t x21 = (1.0 / 6.0) * lambda;
    const real_t x22 = x16 * x21;
    const real_t x23 = x15 * (-x2 * x4 + x7 * x9);
    const real_t x24 = x15 * (x0 * x2 - x6 * x7);
    const real_t x25 = x15 * (-x0 * x9 + x4 * x6);
    const real_t x26 = -x23 - x24 - x25;
    const real_t x27 = u[4] * x26 + u[5] * x23 + u[6] * x24 + u[7] * x25;
    const real_t x28 = x15 * (-x0 * x5 + x7 * x8);
    const real_t x29 = x15 * (x0 * x1 - x4 * x8);
    const real_t x30 = x15 * (-x1 * x7 + x4 * x5);
    const real_t x31 = -x28 - x29 - x30;
    const real_t x32 = u[10] * x28 + u[11] * x29 + u[8] * x31 + u[9] * x30;
    const real_t x33 = u[0] * x31 + u[1] * x30 + u[2] * x28 + u[3] * x29;
    const real_t x34 = (1.0 / 6.0) * mu;
    const real_t x35 = x30 * x34;
    const real_t x36 = u[10] * x17 + u[11] * x18 + u[8] * x19 + u[9] * x16;
    const real_t x37 = u[0] * x26 + u[1] * x23 + u[2] * x24 + u[3] * x25;
    const real_t x38 = x23 * x34;
    const real_t x39 = u[4] * x19 + u[5] * x16 + u[6] * x17 + u[7] * x18;
    const real_t x40 = (1.0 / 3.0) * mu;
    const real_t x41 = x20 * x40;
    const real_t x42 = x16 * x41 + x20 * x22 + x22 * x27 + x22 * x32 + x33 * x35 + x35 * x36 +
                       x37 * x38 + x38 * x39;
    const real_t x43 = x17 * x21;
    const real_t x44 = x28 * x34;
    const real_t x45 = x24 * x34;
    const real_t x46 = x17 * x41 + x20 * x43 + x27 * x43 + x32 * x43 + x33 * x44 + x36 * x44 +
                       x37 * x45 + x39 * x45;
    const real_t x47 = x18 * x20;
    const real_t x48 = x18 * x21;
    const real_t x49 = x29 * x34;
    const real_t x50 = x25 * x34;
    const real_t x51 = x21 * x47 + x27 * x48 + x32 * x48 + x33 * x49 + x36 * x49 + x37 * x50 +
                       x39 * x50 + x40 * x47;
    const real_t x52 = x21 * x23;
    const real_t x53 = u[4] * x31 + u[5] * x30 + u[6] * x28 + u[7] * x29;
    const real_t x54 = u[10] * x24 + u[11] * x25 + u[8] * x26 + u[9] * x23;
    const real_t x55 = x27 * x40;
    const real_t x56 = x16 * x34;
    const real_t x57 = x20 * x52 + x23 * x55 + x27 * x52 + x32 * x52 + x35 * x53 + x35 * x54 +
                       x37 * x56 + x39 * x56;
    const real_t x58 = x21 * x24;
    const real_t x59 = x17 * x34;
    const real_t x60 = x20 * x58 + x24 * x55 + x27 * x58 + x32 * x58 + x37 * x59 + x39 * x59 +
                       x44 * x53 + x44 * x54;
    const real_t x61 = x21 * x25;
    const real_t x62 = x18 * x34;
    const real_t x63 = x20 * x61 + x25 * x55 + x27 * x61 + x32 * x61 + x37 * x62 + x39 * x62 +
                       x49 * x53 + x49 * x54;
    const real_t x64 = x21 * x30;
    const real_t x65 = x32 * x40;
    const real_t x66 = x20 * x64 + x27 * x64 + x30 * x65 + x32 * x64 + x33 * x56 + x36 * x56 +
                       x38 * x53 + x38 * x54;
    const real_t x67 = x21 * x28;
    const real_t x68 = x20 * x67 + x27 * x67 + x28 * x65 + x32 * x67 + x33 * x59 + x36 * x59 +
                       x45 * x53 + x45 * x54;
    const real_t x69 = x21 * x29;
    const real_t x70 = x20 * x69 + x27 * x69 + x29 * x65 + x32 * x69 + x33 * x62 + x36 * x62 +
                       x50 * x53 + x50 * x54;
    element_vector[0 * stride] = x14 * (-x42 - x46 - x51);
    element_vector[1 * stride] = x14 * x42;
    element_vector[2 * stride] = x14 * x46;
    element_vector[3 * stride] = x14 * x51;
    element_vector[4 * stride] = x14 * (-x57 - x60 - x63);
    element_vector[5 * stride] = x14 * x57;
    element_vector[6 * stride] = x14 * x60;
    element_vector[7 * stride] = x14 * x63;
    element_vector[8 * stride] = x14 * (-x66 - x68 - x70);
    element_vector[9 * stride] = x14 * x66;
    element_vector[10 * stride] = x14 * x68;
    element_vector[11 * stride] = x14 * x70;
}

static SFEM_INLINE void tet4_linear_elasticity_assemble_hessian_kernel(const real_t mu,
                                                                       const real_t lambda,
                                                                       const real_t px0,
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
                                                                       real_t *const SFEM_RESTRICT
                                                                           element_matrix) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = -pz0 + pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = -px0 + px2;
    const real_t x5 = -py0 + py3;
    const real_t x6 = -pz0 + pz1;
    const real_t x7 = -px0 + px3;
    const real_t x8 = -py0 + py1;
    const real_t x9 = -pz0 + pz2;
    const real_t x10 = x8 * x9;
    const real_t x11 = x5 * x9;
    const real_t x12 = x2 * x8;
    const real_t x13 = x1 * x6;
    const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x7 - x12 * x4 - x13 * x7 + x4 * x5 * x6;
    const real_t x15 = -x11 + x3;
    const real_t x16 = RPOW2(x14);
    const real_t x17 = x10 - x13;
    const real_t x18 = x16 * x17;
    const real_t x19 = x15 * x18;
    const real_t x20 = (1.0 / 3.0) * lambda;
    const real_t x21 = -x12 + x5 * x6;
    const real_t x22 = x20 * x21;
    const real_t x23 = x15 * x16;
    const real_t x24 = (2.0 / 3.0) * mu;
    const real_t x25 = x21 * x24;
    const real_t x26 = POW2(x15);
    const real_t x27 = (1.0 / 6.0) * lambda;
    const real_t x28 = x16 * x27;
    const real_t x29 = -x1 * x7 + x4 * x5;
    const real_t x30 = POW2(x29);
    const real_t x31 = (1.0 / 6.0) * mu;
    const real_t x32 = x16 * x31;
    const real_t x33 = x30 * x32;
    const real_t x34 = -x2 * x4 + x7 * x9;
    const real_t x35 = POW2(x34);
    const real_t x36 = x32 * x35;
    const real_t x37 = (1.0 / 3.0) * mu;
    const real_t x38 = x16 * x37;
    const real_t x39 = x26 * x28 + x26 * x38 + x33 + x36;
    const real_t x40 = POW2(x21);
    const real_t x41 = -x0 * x5 + x7 * x8;
    const real_t x42 = POW2(x41);
    const real_t x43 = x32 * x42;
    const real_t x44 = x0 * x2 - x6 * x7;
    const real_t x45 = POW2(x44);
    const real_t x46 = x32 * x45;
    const real_t x47 = x28 * x40 + x38 * x40 + x43 + x46;
    const real_t x48 = x16 * POW2(x17);
    const real_t x49 = x0 * x1 - x4 * x8;
    const real_t x50 = POW2(x49);
    const real_t x51 = x32 * x50;
    const real_t x52 = -x0 * x9 + x4 * x6;
    const real_t x53 = POW2(x52);
    const real_t x54 = x32 * x53;
    const real_t x55 = x27 * x48 + x37 * x48 + x51 + x54;
    const real_t x56 = x29 * x41;
    const real_t x57 = x38 * x56;
    const real_t x58 = x38 * x49;
    const real_t x59 = x29 * x58;
    const real_t x60 = x41 * x58;
    const real_t x61 = x57 + x59 + x60;
    const real_t x62 = x38 * x44;
    const real_t x63 = x34 * x62;
    const real_t x64 = x34 * x52;
    const real_t x65 = x38 * x64;
    const real_t x66 = x52 * x62;
    const real_t x67 = x63 + x65 + x66;
    const real_t x68 = x15 * x28;
    const real_t x69 = x32 * x56;
    const real_t x70 = x32 * x44;
    const real_t x71 = x34 * x70;
    const real_t x72 = x15 * x38;
    const real_t x73 = x21 * x72;
    const real_t x74 = x21 * x68 + x69 + x71 + x73;
    const real_t x75 = x32 * x49;
    const real_t x76 = x29 * x75;
    const real_t x77 = x32 * x64;
    const real_t x78 = x17 * x72;
    const real_t x79 = x17 * x68 + x76 + x77 + x78;
    const real_t x80 = x14 * (-x39 - x74 - x79);
    const real_t x81 = x17 * x21;
    const real_t x82 = x41 * x75;
    const real_t x83 = x52 * x70;
    const real_t x84 = x38 * x81;
    const real_t x85 = x28 * x81 + x82 + x83 + x84;
    const real_t x86 = x14 * (-x47 - x74 - x85);
    const real_t x87 = x14 * (-x55 - x79 - x85);
    const real_t x88 = x32 * x34;
    const real_t x89 = x15 * x88 + x34 * x68;
    const real_t x90 = x28 * x34;
    const real_t x91 = x15 * x70 + x21 * x90;
    const real_t x92 = x32 * x52;
    const real_t x93 = x15 * x92 + x17 * x90;
    const real_t x94 = x89 + x91 + x93;
    const real_t x95 = x21 * x88 + x44 * x68;
    const real_t x96 = x28 * x44;
    const real_t x97 = x21 * x70 + x21 * x96;
    const real_t x98 = x17 * x96 + x21 * x92;
    const real_t x99 = x95 + x97 + x98;
    const real_t x100 = x17 * x88 + x52 * x68;
    const real_t x101 = x28 * x52;
    const real_t x102 = x101 * x21 + x17 * x70;
    const real_t x103 = x101 * x17 + x17 * x92;
    const real_t x104 = x100 + x102 + x103;
    const real_t x105 = x14 * (x104 + x94 + x99);
    const real_t x106 = -x14 * x94;
    const real_t x107 = -x14 * x99;
    const real_t x108 = -x104 * x14;
    const real_t x109 = x29 * x32;
    const real_t x110 = x109 * x15 + x29 * x68;
    const real_t x111 = x28 * x29;
    const real_t x112 = x32 * x41;
    const real_t x113 = x111 * x21 + x112 * x15;
    const real_t x114 = x111 * x17 + x15 * x75;
    const real_t x115 = x110 + x113 + x114;
    const real_t x116 = x109 * x21 + x41 * x68;
    const real_t x117 = x28 * x41;
    const real_t x118 = x112 * x21 + x117 * x21;
    const real_t x119 = x117 * x17 + x21 * x75;
    const real_t x120 = x116 + x118 + x119;
    const real_t x121 = x109 * x17 + x49 * x68;
    const real_t x122 = x28 * x49;
    const real_t x123 = x112 * x17 + x122 * x21;
    const real_t x124 = x122 * x17 + x17 * x75;
    const real_t x125 = x121 + x123 + x124;
    const real_t x126 = x14 * (x115 + x120 + x125);
    const real_t x127 = -x115 * x14;
    const real_t x128 = -x120 * x14;
    const real_t x129 = -x125 * x14;
    const real_t x130 = x14 * x74;
    const real_t x131 = x14 * x79;
    const real_t x132 = x14 * (-x100 - x89 - x95);
    const real_t x133 = x14 * x89;
    const real_t x134 = x14 * x95;
    const real_t x135 = x100 * x14;
    const real_t x136 = x14 * (-x110 - x116 - x121);
    const real_t x137 = x110 * x14;
    const real_t x138 = x116 * x14;
    const real_t x139 = x121 * x14;
    const real_t x140 = x14 * x85;
    const real_t x141 = x14 * (-x102 - x91 - x97);
    const real_t x142 = x14 * x91;
    const real_t x143 = x14 * x97;
    const real_t x144 = x102 * x14;
    const real_t x145 = x14 * (-x113 - x118 - x123);
    const real_t x146 = x113 * x14;
    const real_t x147 = x118 * x14;
    const real_t x148 = x123 * x14;
    const real_t x149 = x14 * (-x103 - x93 - x98);
    const real_t x150 = x14 * x93;
    const real_t x151 = x14 * x98;
    const real_t x152 = x103 * x14;
    const real_t x153 = x14 * (-x114 - x119 - x124);
    const real_t x154 = x114 * x14;
    const real_t x155 = x119 * x14;
    const real_t x156 = x124 * x14;
    const real_t x157 = x16 * x20;
    const real_t x158 = x157 * x44;
    const real_t x159 = x16 * x24;
    const real_t x160 = x159 * x44;
    const real_t x161 = x26 * x32;
    const real_t x162 = x161 + x28 * x35 + x33 + x35 * x38;
    const real_t x163 = x32 * x40;
    const real_t x164 = x163 + x28 * x45 + x38 * x45 + x43;
    const real_t x165 = x31 * x48;
    const real_t x166 = x165 + x28 * x53 + x38 * x53 + x51;
    const real_t x167 = x73 + x78 + x84;
    const real_t x168 = x15 * x32;
    const real_t x169 = x168 * x21;
    const real_t x170 = x169 + x34 * x96 + x63 + x69;
    const real_t x171 = x168 * x17;
    const real_t x172 = x171 + x28 * x64 + x65 + x76;
    const real_t x173 = x14 * (-x162 - x170 - x172);
    const real_t x174 = x32 * x81;
    const real_t x175 = x174 + x52 * x96 + x66 + x82;
    const real_t x176 = x14 * (-x164 - x170 - x175);
    const real_t x177 = x14 * (-x166 - x172 - x175);
    const real_t x178 = x29 * x88 + x29 * x90;
    const real_t x179 = x29 * x96 + x41 * x88;
    const real_t x180 = x101 * x29 + x34 * x75;
    const real_t x181 = x178 + x179 + x180;
    const real_t x182 = x29 * x70 + x41 * x90;
    const real_t x183 = x41 * x70 + x41 * x96;
    const real_t x184 = x101 * x41 + x44 * x75;
    const real_t x185 = x182 + x183 + x184;
    const real_t x186 = x29 * x92 + x49 * x90;
    const real_t x187 = x41 * x92 + x49 * x96;
    const real_t x188 = x101 * x49 + x52 * x75;
    const real_t x189 = x186 + x187 + x188;
    const real_t x190 = x14 * (x181 + x185 + x189);
    const real_t x191 = -x14 * x181;
    const real_t x192 = -x14 * x185;
    const real_t x193 = -x14 * x189;
    const real_t x194 = x14 * x170;
    const real_t x195 = x14 * x172;
    const real_t x196 = x14 * (-x178 - x182 - x186);
    const real_t x197 = x14 * x178;
    const real_t x198 = x14 * x182;
    const real_t x199 = x14 * x186;
    const real_t x200 = x14 * x175;
    const real_t x201 = x14 * (-x179 - x183 - x187);
    const real_t x202 = x14 * x179;
    const real_t x203 = x14 * x183;
    const real_t x204 = x14 * x187;
    const real_t x205 = x14 * (-x180 - x184 - x188);
    const real_t x206 = x14 * x180;
    const real_t x207 = x14 * x184;
    const real_t x208 = x14 * x188;
    const real_t x209 = x157 * x49;
    const real_t x210 = x159 * x49;
    const real_t x211 = x161 + x28 * x30 + x30 * x38 + x36;
    const real_t x212 = x163 + x28 * x42 + x38 * x42 + x46;
    const real_t x213 = x165 + x28 * x50 + x38 * x50 + x54;
    const real_t x214 = x169 + x28 * x56 + x57 + x71;
    const real_t x215 = x122 * x29 + x171 + x59 + x77;
    const real_t x216 = x14 * (-x211 - x214 - x215);
    const real_t x217 = x122 * x41 + x174 + x60 + x83;
    const real_t x218 = x14 * (-x212 - x214 - x217);
    const real_t x219 = x14 * (-x213 - x215 - x217);
    const real_t x220 = x14 * x214;
    const real_t x221 = x14 * x215;
    const real_t x222 = x14 * x217;
    element_matrix[0 * stride] = x14 * (x18 * x22 + x18 * x25 + x19 * x20 + x19 * x24 + x22 * x23 +
                                        x23 * x25 + x39 + x47 + x55 + x61 + x67);
    element_matrix[1 * stride] = x80;
    element_matrix[2 * stride] = x86;
    element_matrix[3 * stride] = x87;
    element_matrix[4 * stride] = x105;
    element_matrix[5 * stride] = x106;
    element_matrix[6 * stride] = x107;
    element_matrix[7 * stride] = x108;
    element_matrix[8 * stride] = x126;
    element_matrix[9 * stride] = x127;
    element_matrix[10 * stride] = x128;
    element_matrix[11 * stride] = x129;
    element_matrix[12 * stride] = x80;
    element_matrix[13 * stride] = x14 * x39;
    element_matrix[14 * stride] = x130;
    element_matrix[15 * stride] = x131;
    element_matrix[16 * stride] = x132;
    element_matrix[17 * stride] = x133;
    element_matrix[18 * stride] = x134;
    element_matrix[19 * stride] = x135;
    element_matrix[20 * stride] = x136;
    element_matrix[21 * stride] = x137;
    element_matrix[22 * stride] = x138;
    element_matrix[23 * stride] = x139;
    element_matrix[24 * stride] = x86;
    element_matrix[25 * stride] = x130;
    element_matrix[26 * stride] = x14 * x47;
    element_matrix[27 * stride] = x140;
    element_matrix[28 * stride] = x141;
    element_matrix[29 * stride] = x142;
    element_matrix[30 * stride] = x143;
    element_matrix[31 * stride] = x144;
    element_matrix[32 * stride] = x145;
    element_matrix[33 * stride] = x146;
    element_matrix[34 * stride] = x147;
    element_matrix[35 * stride] = x148;
    element_matrix[36 * stride] = x87;
    element_matrix[37 * stride] = x131;
    element_matrix[38 * stride] = x140;
    element_matrix[39 * stride] = x14 * x55;
    element_matrix[40 * stride] = x149;
    element_matrix[41 * stride] = x150;
    element_matrix[42 * stride] = x151;
    element_matrix[43 * stride] = x152;
    element_matrix[44 * stride] = x153;
    element_matrix[45 * stride] = x154;
    element_matrix[46 * stride] = x155;
    element_matrix[47 * stride] = x156;
    element_matrix[48 * stride] = x105;
    element_matrix[49 * stride] = x132;
    element_matrix[50 * stride] = x141;
    element_matrix[51 * stride] = x149;
    element_matrix[52 * stride] = x14 * (x157 * x64 + x158 * x34 + x158 * x52 + x159 * x64 +
                                         x160 * x34 + x160 * x52 + x162 + x164 + x166 + x167 + x61);
    element_matrix[53 * stride] = x173;
    element_matrix[54 * stride] = x176;
    element_matrix[55 * stride] = x177;
    element_matrix[56 * stride] = x190;
    element_matrix[57 * stride] = x191;
    element_matrix[58 * stride] = x192;
    element_matrix[59 * stride] = x193;
    element_matrix[60 * stride] = x106;
    element_matrix[61 * stride] = x133;
    element_matrix[62 * stride] = x142;
    element_matrix[63 * stride] = x150;
    element_matrix[64 * stride] = x173;
    element_matrix[65 * stride] = x14 * x162;
    element_matrix[66 * stride] = x194;
    element_matrix[67 * stride] = x195;
    element_matrix[68 * stride] = x196;
    element_matrix[69 * stride] = x197;
    element_matrix[70 * stride] = x198;
    element_matrix[71 * stride] = x199;
    element_matrix[72 * stride] = x107;
    element_matrix[73 * stride] = x134;
    element_matrix[74 * stride] = x143;
    element_matrix[75 * stride] = x151;
    element_matrix[76 * stride] = x176;
    element_matrix[77 * stride] = x194;
    element_matrix[78 * stride] = x14 * x164;
    element_matrix[79 * stride] = x200;
    element_matrix[80 * stride] = x201;
    element_matrix[81 * stride] = x202;
    element_matrix[82 * stride] = x203;
    element_matrix[83 * stride] = x204;
    element_matrix[84 * stride] = x108;
    element_matrix[85 * stride] = x135;
    element_matrix[86 * stride] = x144;
    element_matrix[87 * stride] = x152;
    element_matrix[88 * stride] = x177;
    element_matrix[89 * stride] = x195;
    element_matrix[90 * stride] = x200;
    element_matrix[91 * stride] = x14 * x166;
    element_matrix[92 * stride] = x205;
    element_matrix[93 * stride] = x206;
    element_matrix[94 * stride] = x207;
    element_matrix[95 * stride] = x208;
    element_matrix[96 * stride] = x126;
    element_matrix[97 * stride] = x136;
    element_matrix[98 * stride] = x145;
    element_matrix[99 * stride] = x153;
    element_matrix[100 * stride] = x190;
    element_matrix[101 * stride] = x196;
    element_matrix[102 * stride] = x201;
    element_matrix[103 * stride] = x205;
    element_matrix[104 * stride] = x14 * (x157 * x56 + x159 * x56 + x167 + x209 * x29 + x209 * x41 +
                                          x210 * x29 + x210 * x41 + x211 + x212 + x213 + x67);
    element_matrix[105 * stride] = x216;
    element_matrix[106 * stride] = x218;
    element_matrix[107 * stride] = x219;
    element_matrix[108 * stride] = x127;
    element_matrix[109 * stride] = x137;
    element_matrix[110 * stride] = x146;
    element_matrix[111 * stride] = x154;
    element_matrix[112 * stride] = x191;
    element_matrix[113 * stride] = x197;
    element_matrix[114 * stride] = x202;
    element_matrix[115 * stride] = x206;
    element_matrix[116 * stride] = x216;
    element_matrix[117 * stride] = x14 * x211;
    element_matrix[118 * stride] = x220;
    element_matrix[119 * stride] = x221;
    element_matrix[120 * stride] = x128;
    element_matrix[121 * stride] = x138;
    element_matrix[122 * stride] = x147;
    element_matrix[123 * stride] = x155;
    element_matrix[124 * stride] = x192;
    element_matrix[125 * stride] = x198;
    element_matrix[126 * stride] = x203;
    element_matrix[127 * stride] = x207;
    element_matrix[128 * stride] = x218;
    element_matrix[129 * stride] = x220;
    element_matrix[130 * stride] = x14 * x212;
    element_matrix[131 * stride] = x222;
    element_matrix[132 * stride] = x129;
    element_matrix[133 * stride] = x139;
    element_matrix[134 * stride] = x148;
    element_matrix[135 * stride] = x156;
    element_matrix[136 * stride] = x193;
    element_matrix[137 * stride] = x199;
    element_matrix[138 * stride] = x204;
    element_matrix[139 * stride] = x208;
    element_matrix[140 * stride] = x219;
    element_matrix[141 * stride] = x221;
    element_matrix[142 * stride] = x222;
    element_matrix[143 * stride] = x14 * x213;
}

#ifdef SFEM_ENABLE_EXPLICIT_VECTORIZATION

static SFEM_INLINE void tet4_linear_elasticity_apply_kernel(
    const vreal_t mu,
    const vreal_t lambda,
    const vreal_t px0,
    const vreal_t px1,
    const vreal_t px2,
    const vreal_t px3,
    const vreal_t py0,
    const vreal_t py1,
    const vreal_t py2,
    const vreal_t py3,
    const vreal_t pz0,
    const vreal_t pz1,
    const vreal_t pz2,
    const vreal_t pz3,
    const vreal_t *const SFEM_RESTRICT increment,
    vreal_t *const SFEM_RESTRICT element_vector) {
    const vreal_t x0 = -py0 + py2;
    const vreal_t x1 = -pz0 + pz3;
    const vreal_t x2 = x0 * x1;
    const vreal_t x3 = -py0 + py3;
    const vreal_t x4 = -pz0 + pz2;
    const vreal_t x5 = x3 * x4;
    const vreal_t x6 = x2 - x5;
    const vreal_t x7 = -px0 + px1;
    const vreal_t x8 = -px0 + px3;
    const vreal_t x9 = -py0 + py1;
    const vreal_t x10 = -x3 * x7 + x8 * x9;
    const vreal_t x11 = -px0 + px2;
    const vreal_t x12 = -pz0 + pz1;
    const vreal_t x13 = x4 * x9;
    const vreal_t x14 = x1 * x9;
    const vreal_t x15 = x0 * x12;
    const vreal_t x16 = x11 * x12 * x3 - x11 * x14 + x13 * x8 - x15 * x8 + x2 * x7 - x5 * x7;
    const vreal_t x17 = RPOW2(x16);
    const vreal_t x18 = (1.0 / 6.0) * lambda;
    const vreal_t x19 = x17 * x18;
    const vreal_t x20 = x10 * x19;
    const vreal_t x21 = -x0 * x8 + x11 * x3;
    const vreal_t x22 = x12 * x3 - x14;
    const vreal_t x23 = (1.0 / 6.0) * mu;
    const vreal_t x24 = x17 * x23;
    const vreal_t x25 = x22 * x24;
    const vreal_t x26 = x20 * x6 + x21 * x25;
    const vreal_t x27 = x10 * x25 + x20 * x22;
    const vreal_t x28 = x13 - x15;
    const vreal_t x29 = x0 * x7 - x11 * x9;
    const vreal_t x30 = x20 * x28 + x25 * x29;
    const vreal_t x31 = x26 + x27 + x30;
    const vreal_t x32 = -x31;
    const vreal_t x33 = increment[10] * x16;
    const vreal_t x34 = x19 * x29;
    const vreal_t x35 = x24 * x28;
    const vreal_t x36 = x21 * x35 + x34 * x6;
    const vreal_t x37 = x10 * x35 + x22 * x34;
    const vreal_t x38 = x28 * x34 + x29 * x35;
    const vreal_t x39 = x36 + x37 + x38;
    const vreal_t x40 = -x39;
    const vreal_t x41 = increment[11] * x16;
    const vreal_t x42 = -x1 * x11 + x4 * x8;
    const vreal_t x43 = x19 * x6;
    const vreal_t x44 = x24 * x6;
    const vreal_t x45 = x42 * x43 + x42 * x44;
    const vreal_t x46 = x19 * x42;
    const vreal_t x47 = x1 * x7 - x12 * x8;
    const vreal_t x48 = x24 * x47;
    const vreal_t x49 = x22 * x46 + x48 * x6;
    const vreal_t x50 = x11 * x12 - x4 * x7;
    const vreal_t x51 = x28 * x46 + x44 * x50;
    const vreal_t x52 = x45 + x49 + x51;
    const vreal_t x53 = -x52;
    const vreal_t x54 = increment[5] * x16;
    const vreal_t x55 = x25 * x42 + x43 * x47;
    const vreal_t x56 = x19 * x47;
    const vreal_t x57 = x22 * x56 + x25 * x47;
    const vreal_t x58 = x25 * x50 + x28 * x56;
    const vreal_t x59 = x55 + x57 + x58;
    const vreal_t x60 = -x59;
    const vreal_t x61 = increment[6] * x16;
    const vreal_t x62 = x35 * x42 + x43 * x50;
    const vreal_t x63 = x19 * x50;
    const vreal_t x64 = x22 * x63 + x35 * x47;
    const vreal_t x65 = x28 * x63 + x35 * x50;
    const vreal_t x66 = x62 + x64 + x65;
    const vreal_t x67 = -x66;
    const vreal_t x68 = increment[7] * x16;
    const vreal_t x69 = x21 * x43 + x21 * x44;
    const vreal_t x70 = x19 * x21;
    const vreal_t x71 = x10 * x44 + x22 * x70;
    const vreal_t x72 = x24 * x29;
    const vreal_t x73 = x28 * x70 + x6 * x72;
    const vreal_t x74 = x69 + x71 + x73;
    const vreal_t x75 = -x74;
    const vreal_t x76 = increment[9] * x16;
    const vreal_t x77 = POW2(x6);
    const vreal_t x78 = POW2(x21);
    const vreal_t x79 = x24 * x78;
    const vreal_t x80 = POW2(x42);
    const vreal_t x81 = x24 * x80;
    const vreal_t x82 = (1.0 / 3.0) * mu;
    const vreal_t x83 = x17 * x82;
    const vreal_t x84 = x19 * x77 + x77 * x83 + x79 + x81;
    const vreal_t x85 = x10 * x21;
    const vreal_t x86 = x24 * x85;
    const vreal_t x87 = x42 * x48;
    const vreal_t x88 = x6 * x83;
    const vreal_t x89 = x22 * x88;
    const vreal_t x90 = x22 * x43 + x86 + x87 + x89;
    const vreal_t x91 = x21 * x72;
    const vreal_t x92 = x42 * x50;
    const vreal_t x93 = x24 * x92;
    const vreal_t x94 = x28 * x88;
    const vreal_t x95 = x28 * x43 + x91 + x93 + x94;
    const vreal_t x96 = -x84 - x90 - x95;
    const vreal_t x97 = increment[1] * x16;
    const vreal_t x98 = POW2(x22);
    const vreal_t x99 = POW2(x10);
    const vreal_t x100 = x24 * x99;
    const vreal_t x101 = POW2(x47);
    const vreal_t x102 = x101 * x24;
    const vreal_t x103 = x100 + x102 + x19 * x98 + x83 * x98;
    const vreal_t x104 = x22 * x28;
    const vreal_t x105 = x10 * x72;
    const vreal_t x106 = x48 * x50;
    const vreal_t x107 = x104 * x83;
    const vreal_t x108 = x104 * x19 + x105 + x106 + x107;
    const vreal_t x109 = -x103 - x108 - x90;
    const vreal_t x110 = increment[2] * x16;
    const vreal_t x111 = x17 * POW2(x28);
    const vreal_t x112 = POW2(x29);
    const vreal_t x113 = x112 * x24;
    const vreal_t x114 = POW2(x50);
    const vreal_t x115 = x114 * x24;
    const vreal_t x116 = x111 * x18 + x111 * x82 + x113 + x115;
    const vreal_t x117 = -x108 - x116 - x95;
    const vreal_t x118 = increment[3] * x16;
    const vreal_t x119 = x52 + x59 + x66;
    const vreal_t x120 = increment[4] * x16;
    const vreal_t x121 = x31 + x39 + x74;
    const vreal_t x122 = increment[8] * x16;
    const vreal_t x123 = x17 * x28;
    const vreal_t x124 = x123 * x6;
    const vreal_t x125 = (1.0 / 3.0) * lambda;
    const vreal_t x126 = x125 * x22;
    const vreal_t x127 = x17 * x6;
    const vreal_t x128 = (2.0 / 3.0) * mu;
    const vreal_t x129 = x128 * x22;
    const vreal_t x130 = x83 * x85;
    const vreal_t x131 = x29 * x83;
    const vreal_t x132 = x131 * x21;
    const vreal_t x133 = x10 * x131;
    const vreal_t x134 = x130 + x132 + x133;
    const vreal_t x135 = x47 * x83;
    const vreal_t x136 = x135 * x42;
    const vreal_t x137 = x83 * x92;
    const vreal_t x138 = x135 * x50;
    const vreal_t x139 = x136 + x137 + x138;
    const vreal_t x140 = increment[0] * x16;
    const vreal_t x141 = -x45 - x55 - x62;
    const vreal_t x142 = -x26 - x36 - x69;
    const vreal_t x143 = -x49 - x57 - x64;
    const vreal_t x144 = -x27 - x37 - x71;
    const vreal_t x145 = -x51 - x58 - x65;
    const vreal_t x146 = -x30 - x38 - x73;
    const vreal_t x147 = x20 * x42 + x21 * x48;
    const vreal_t x148 = x10 * x48 + x20 * x47;
    const vreal_t x149 = x20 * x50 + x47 * x72;
    const vreal_t x150 = x147 + x148 + x149;
    const vreal_t x151 = -x150;
    const vreal_t x152 = x24 * x50;
    const vreal_t x153 = x152 * x21 + x34 * x42;
    const vreal_t x154 = x10 * x152 + x34 * x47;
    const vreal_t x155 = x34 * x50 + x50 * x72;
    const vreal_t x156 = x153 + x154 + x155;
    const vreal_t x157 = -x156;
    const vreal_t x158 = x24 * x42;
    const vreal_t x159 = x158 * x21 + x21 * x46;
    const vreal_t x160 = x10 * x158 + x21 * x56;
    const vreal_t x161 = x21 * x63 + x42 * x72;
    const vreal_t x162 = x159 + x160 + x161;
    const vreal_t x163 = -x162;
    const vreal_t x164 = x24 * x77;
    const vreal_t x165 = x164 + x19 * x80 + x79 + x80 * x83;
    const vreal_t x166 = x25 * x6;
    const vreal_t x167 = x136 + x166 + x42 * x56 + x86;
    const vreal_t x168 = x35 * x6;
    const vreal_t x169 = x137 + x168 + x19 * x92 + x91;
    const vreal_t x170 = -x165 - x167 - x169;
    const vreal_t x171 = x24 * x98;
    const vreal_t x172 = x100 + x101 * x19 + x101 * x83 + x171;
    const vreal_t x173 = x25 * x28;
    const vreal_t x174 = x105 + x138 + x173 + x50 * x56;
    const vreal_t x175 = -x167 - x172 - x174;
    const vreal_t x176 = x111 * x23;
    const vreal_t x177 = x113 + x114 * x19 + x114 * x83 + x176;
    const vreal_t x178 = -x169 - x174 - x177;
    const vreal_t x179 = x150 + x156 + x162;
    const vreal_t x180 = x125 * x17;
    const vreal_t x181 = x180 * x47;
    const vreal_t x182 = x128 * x17;
    const vreal_t x183 = x182 * x47;
    const vreal_t x184 = x107 + x89 + x94;
    const vreal_t x185 = -x147 - x153 - x159;
    const vreal_t x186 = -x148 - x154 - x160;
    const vreal_t x187 = -x149 - x155 - x161;
    const vreal_t x188 = x130 + x166 + x19 * x85 + x87;
    const vreal_t x189 = x102 + x171 + x19 * x99 + x83 * x99;
    const vreal_t x190 = x106 + x133 + x173 + x20 * x29;
    const vreal_t x191 = -x188 - x189 - x190;
    const vreal_t x192 = x132 + x168 + x21 * x34 + x93;
    const vreal_t x193 = x112 * x19 + x112 * x83 + x115 + x176;
    const vreal_t x194 = -x190 - x192 - x193;
    const vreal_t x195 = x164 + x19 * x78 + x78 * x83 + x81;
    const vreal_t x196 = -x188 - x192 - x195;
    const vreal_t x197 = x180 * x29;
    const vreal_t x198 = x182 * x29;
    element_vector[0 * stride] =
        x109 * x110 + x117 * x118 + x119 * x120 + x121 * x122 +
        x140 * (x103 + x116 + x123 * x126 + x123 * x129 + x124 * x125 + x124 * x128 + x126 * x127 +
                x127 * x129 + x134 + x139 + x84) +
        x32 * x33 + x40 * x41 + x53 * x54 + x60 * x61 + x67 * x68 + x75 * x76 + x96 * x97;
    element_vector[1 * stride] = x110 * x90 + x118 * x95 + x120 * x141 + x122 * x142 + x140 * x96 +
                                 x26 * x33 + x36 * x41 + x45 * x54 + x55 * x61 + x62 * x68 +
                                 x69 * x76 + x84 * x97;
    element_vector[2 * stride] = x103 * x110 + x108 * x118 + x109 * x140 + x120 * x143 +
                                 x122 * x144 + x27 * x33 + x37 * x41 + x49 * x54 + x57 * x61 +
                                 x64 * x68 + x71 * x76 + x90 * x97;
    element_vector[3 * stride] = x108 * x110 + x116 * x118 + x117 * x140 + x120 * x145 +
                                 x122 * x146 + x30 * x33 + x38 * x41 + x51 * x54 + x58 * x61 +
                                 x65 * x68 + x73 * x76 + x95 * x97;
    element_vector[4 * stride] = x110 * x143 + x118 * x145 + x119 * x140 +
                                 x120 * (x134 + x165 + x172 + x177 + x180 * x92 + x181 * x42 +
                                         x181 * x50 + x182 * x92 + x183 * x42 + x183 * x50 + x184) +
                                 x122 * x179 + x141 * x97 + x151 * x33 + x157 * x41 + x163 * x76 +
                                 x170 * x54 + x175 * x61 + x178 * x68;
    element_vector[5 * stride] = x110 * x49 + x118 * x51 + x120 * x170 + x122 * x185 + x140 * x53 +
                                 x147 * x33 + x153 * x41 + x159 * x76 + x165 * x54 + x167 * x61 +
                                 x169 * x68 + x45 * x97;
    element_vector[6 * stride] = x110 * x57 + x118 * x58 + x120 * x175 + x122 * x186 + x140 * x60 +
                                 x148 * x33 + x154 * x41 + x160 * x76 + x167 * x54 + x172 * x61 +
                                 x174 * x68 + x55 * x97;
    element_vector[7 * stride] = x110 * x64 + x118 * x65 + x120 * x178 + x122 * x187 + x140 * x67 +
                                 x149 * x33 + x155 * x41 + x161 * x76 + x169 * x54 + x174 * x61 +
                                 x177 * x68 + x62 * x97;
    element_vector[8 * stride] = x110 * x144 + x118 * x146 + x120 * x179 + x121 * x140 +
                                 x122 * (x10 * x197 + x10 * x198 + x139 + x180 * x85 + x182 * x85 +
                                         x184 + x189 + x193 + x195 + x197 * x21 + x198 * x21) +
                                 x142 * x97 + x185 * x54 + x186 * x61 + x187 * x68 + x191 * x33 +
                                 x194 * x41 + x196 * x76;
    element_vector[9 * stride] = x110 * x71 + x118 * x73 + x120 * x163 + x122 * x196 + x140 * x75 +
                                 x159 * x54 + x160 * x61 + x161 * x68 + x188 * x33 + x192 * x41 +
                                 x195 * x76 + x69 * x97;
    element_vector[10 * stride] = x110 * x27 + x118 * x30 + x120 * x151 + x122 * x191 + x140 * x32 +
                                  x147 * x54 + x148 * x61 + x149 * x68 + x188 * x76 + x189 * x33 +
                                  x190 * x41 + x26 * x97;
    element_vector[11 * stride] = x110 * x37 + x118 * x38 + x120 * x157 + x122 * x194 + x140 * x40 +
                                  x153 * x54 + x154 * x61 + x155 * x68 + x190 * x33 + x192 * x76 +
                                  x193 * x41 + x36 * x97;
}

#else

static SFEM_INLINE void tet4_linear_elasticity_apply_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
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
    const real_t *const SFEM_RESTRICT increment,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = -px0 + px1;
    const real_t x8 = -px0 + px3;
    const real_t x9 = -py0 + py1;
    const real_t x10 = -x3 * x7 + x8 * x9;
    const real_t x11 = -px0 + px2;
    const real_t x12 = -pz0 + pz1;
    const real_t x13 = x4 * x9;
    const real_t x14 = x1 * x9;
    const real_t x15 = x0 * x12;
    const real_t x16 = x11 * x12 * x3 - x11 * x14 + x13 * x8 - x15 * x8 + x2 * x7 - x5 * x7;
    const real_t x17 = RPOW2(x16);
    const real_t x18 = (1.0 / 6.0) * lambda;
    const real_t x19 = x17 * x18;
    const real_t x20 = x10 * x19;
    const real_t x21 = -x0 * x8 + x11 * x3;
    const real_t x22 = x12 * x3 - x14;
    const real_t x23 = (1.0 / 6.0) * mu;
    const real_t x24 = x17 * x23;
    const real_t x25 = x22 * x24;
    const real_t x26 = x20 * x6 + x21 * x25;
    const real_t x27 = x10 * x25 + x20 * x22;
    const real_t x28 = x13 - x15;
    const real_t x29 = x0 * x7 - x11 * x9;
    const real_t x30 = x20 * x28 + x25 * x29;
    const real_t x31 = x26 + x27 + x30;
    const real_t x32 = -x31;
    const real_t x33 = increment[10] * x16;
    const real_t x34 = x19 * x29;
    const real_t x35 = x24 * x28;
    const real_t x36 = x21 * x35 + x34 * x6;
    const real_t x37 = x10 * x35 + x22 * x34;
    const real_t x38 = x28 * x34 + x29 * x35;
    const real_t x39 = x36 + x37 + x38;
    const real_t x40 = -x39;
    const real_t x41 = increment[11] * x16;
    const real_t x42 = -x1 * x11 + x4 * x8;
    const real_t x43 = x19 * x6;
    const real_t x44 = x24 * x6;
    const real_t x45 = x42 * x43 + x42 * x44;
    const real_t x46 = x19 * x42;
    const real_t x47 = x1 * x7 - x12 * x8;
    const real_t x48 = x24 * x47;
    const real_t x49 = x22 * x46 + x48 * x6;
    const real_t x50 = x11 * x12 - x4 * x7;
    const real_t x51 = x28 * x46 + x44 * x50;
    const real_t x52 = x45 + x49 + x51;
    const real_t x53 = -x52;
    const real_t x54 = increment[5] * x16;
    const real_t x55 = x25 * x42 + x43 * x47;
    const real_t x56 = x19 * x47;
    const real_t x57 = x22 * x56 + x25 * x47;
    const real_t x58 = x25 * x50 + x28 * x56;
    const real_t x59 = x55 + x57 + x58;
    const real_t x60 = -x59;
    const real_t x61 = increment[6] * x16;
    const real_t x62 = x35 * x42 + x43 * x50;
    const real_t x63 = x19 * x50;
    const real_t x64 = x22 * x63 + x35 * x47;
    const real_t x65 = x28 * x63 + x35 * x50;
    const real_t x66 = x62 + x64 + x65;
    const real_t x67 = -x66;
    const real_t x68 = increment[7] * x16;
    const real_t x69 = x21 * x43 + x21 * x44;
    const real_t x70 = x19 * x21;
    const real_t x71 = x10 * x44 + x22 * x70;
    const real_t x72 = x24 * x29;
    const real_t x73 = x28 * x70 + x6 * x72;
    const real_t x74 = x69 + x71 + x73;
    const real_t x75 = -x74;
    const real_t x76 = increment[9] * x16;
    const real_t x77 = POW2(x6);
    const real_t x78 = POW2(x21);
    const real_t x79 = x24 * x78;
    const real_t x80 = POW2(x42);
    const real_t x81 = x24 * x80;
    const real_t x82 = (1.0 / 3.0) * mu;
    const real_t x83 = x17 * x82;
    const real_t x84 = x19 * x77 + x77 * x83 + x79 + x81;
    const real_t x85 = x10 * x21;
    const real_t x86 = x24 * x85;
    const real_t x87 = x42 * x48;
    const real_t x88 = x6 * x83;
    const real_t x89 = x22 * x88;
    const real_t x90 = x22 * x43 + x86 + x87 + x89;
    const real_t x91 = x21 * x72;
    const real_t x92 = x42 * x50;
    const real_t x93 = x24 * x92;
    const real_t x94 = x28 * x88;
    const real_t x95 = x28 * x43 + x91 + x93 + x94;
    const real_t x96 = -x84 - x90 - x95;
    const real_t x97 = increment[1] * x16;
    const real_t x98 = POW2(x22);
    const real_t x99 = POW2(x10);
    const real_t x100 = x24 * x99;
    const real_t x101 = POW2(x47);
    const real_t x102 = x101 * x24;
    const real_t x103 = x100 + x102 + x19 * x98 + x83 * x98;
    const real_t x104 = x22 * x28;
    const real_t x105 = x10 * x72;
    const real_t x106 = x48 * x50;
    const real_t x107 = x104 * x83;
    const real_t x108 = x104 * x19 + x105 + x106 + x107;
    const real_t x109 = -x103 - x108 - x90;
    const real_t x110 = increment[2] * x16;
    const real_t x111 = x17 * POW2(x28);
    const real_t x112 = POW2(x29);
    const real_t x113 = x112 * x24;
    const real_t x114 = POW2(x50);
    const real_t x115 = x114 * x24;
    const real_t x116 = x111 * x18 + x111 * x82 + x113 + x115;
    const real_t x117 = -x108 - x116 - x95;
    const real_t x118 = increment[3] * x16;
    const real_t x119 = x52 + x59 + x66;
    const real_t x120 = increment[4] * x16;
    const real_t x121 = x31 + x39 + x74;
    const real_t x122 = increment[8] * x16;
    const real_t x123 = x17 * x28;
    const real_t x124 = x123 * x6;
    const real_t x125 = (1.0 / 3.0) * lambda;
    const real_t x126 = x125 * x22;
    const real_t x127 = x17 * x6;
    const real_t x128 = (2.0 / 3.0) * mu;
    const real_t x129 = x128 * x22;
    const real_t x130 = x83 * x85;
    const real_t x131 = x29 * x83;
    const real_t x132 = x131 * x21;
    const real_t x133 = x10 * x131;
    const real_t x134 = x130 + x132 + x133;
    const real_t x135 = x47 * x83;
    const real_t x136 = x135 * x42;
    const real_t x137 = x83 * x92;
    const real_t x138 = x135 * x50;
    const real_t x139 = x136 + x137 + x138;
    const real_t x140 = increment[0] * x16;
    const real_t x141 = -x45 - x55 - x62;
    const real_t x142 = -x26 - x36 - x69;
    const real_t x143 = -x49 - x57 - x64;
    const real_t x144 = -x27 - x37 - x71;
    const real_t x145 = -x51 - x58 - x65;
    const real_t x146 = -x30 - x38 - x73;
    const real_t x147 = x20 * x42 + x21 * x48;
    const real_t x148 = x10 * x48 + x20 * x47;
    const real_t x149 = x20 * x50 + x47 * x72;
    const real_t x150 = x147 + x148 + x149;
    const real_t x151 = -x150;
    const real_t x152 = x24 * x50;
    const real_t x153 = x152 * x21 + x34 * x42;
    const real_t x154 = x10 * x152 + x34 * x47;
    const real_t x155 = x34 * x50 + x50 * x72;
    const real_t x156 = x153 + x154 + x155;
    const real_t x157 = -x156;
    const real_t x158 = x24 * x42;
    const real_t x159 = x158 * x21 + x21 * x46;
    const real_t x160 = x10 * x158 + x21 * x56;
    const real_t x161 = x21 * x63 + x42 * x72;
    const real_t x162 = x159 + x160 + x161;
    const real_t x163 = -x162;
    const real_t x164 = x24 * x77;
    const real_t x165 = x164 + x19 * x80 + x79 + x80 * x83;
    const real_t x166 = x25 * x6;
    const real_t x167 = x136 + x166 + x42 * x56 + x86;
    const real_t x168 = x35 * x6;
    const real_t x169 = x137 + x168 + x19 * x92 + x91;
    const real_t x170 = -x165 - x167 - x169;
    const real_t x171 = x24 * x98;
    const real_t x172 = x100 + x101 * x19 + x101 * x83 + x171;
    const real_t x173 = x25 * x28;
    const real_t x174 = x105 + x138 + x173 + x50 * x56;
    const real_t x175 = -x167 - x172 - x174;
    const real_t x176 = x111 * x23;
    const real_t x177 = x113 + x114 * x19 + x114 * x83 + x176;
    const real_t x178 = -x169 - x174 - x177;
    const real_t x179 = x150 + x156 + x162;
    const real_t x180 = x125 * x17;
    const real_t x181 = x180 * x47;
    const real_t x182 = x128 * x17;
    const real_t x183 = x182 * x47;
    const real_t x184 = x107 + x89 + x94;
    const real_t x185 = -x147 - x153 - x159;
    const real_t x186 = -x148 - x154 - x160;
    const real_t x187 = -x149 - x155 - x161;
    const real_t x188 = x130 + x166 + x19 * x85 + x87;
    const real_t x189 = x102 + x171 + x19 * x99 + x83 * x99;
    const real_t x190 = x106 + x133 + x173 + x20 * x29;
    const real_t x191 = -x188 - x189 - x190;
    const real_t x192 = x132 + x168 + x21 * x34 + x93;
    const real_t x193 = x112 * x19 + x112 * x83 + x115 + x176;
    const real_t x194 = -x190 - x192 - x193;
    const real_t x195 = x164 + x19 * x78 + x78 * x83 + x81;
    const real_t x196 = -x188 - x192 - x195;
    const real_t x197 = x180 * x29;
    const real_t x198 = x182 * x29;
    element_vector[0 * stride] =
        x109 * x110 + x117 * x118 + x119 * x120 + x121 * x122 +
        x140 * (x103 + x116 + x123 * x126 + x123 * x129 + x124 * x125 + x124 * x128 + x126 * x127 +
                x127 * x129 + x134 + x139 + x84) +
        x32 * x33 + x40 * x41 + x53 * x54 + x60 * x61 + x67 * x68 + x75 * x76 + x96 * x97;
    element_vector[1 * stride] = x110 * x90 + x118 * x95 + x120 * x141 + x122 * x142 + x140 * x96 +
                                 x26 * x33 + x36 * x41 + x45 * x54 + x55 * x61 + x62 * x68 +
                                 x69 * x76 + x84 * x97;
    element_vector[2 * stride] = x103 * x110 + x108 * x118 + x109 * x140 + x120 * x143 +
                                 x122 * x144 + x27 * x33 + x37 * x41 + x49 * x54 + x57 * x61 +
                                 x64 * x68 + x71 * x76 + x90 * x97;
    element_vector[3 * stride] = x108 * x110 + x116 * x118 + x117 * x140 + x120 * x145 +
                                 x122 * x146 + x30 * x33 + x38 * x41 + x51 * x54 + x58 * x61 +
                                 x65 * x68 + x73 * x76 + x95 * x97;
    element_vector[4 * stride] = x110 * x143 + x118 * x145 + x119 * x140 +
                                 x120 * (x134 + x165 + x172 + x177 + x180 * x92 + x181 * x42 +
                                         x181 * x50 + x182 * x92 + x183 * x42 + x183 * x50 + x184) +
                                 x122 * x179 + x141 * x97 + x151 * x33 + x157 * x41 + x163 * x76 +
                                 x170 * x54 + x175 * x61 + x178 * x68;
    element_vector[5 * stride] = x110 * x49 + x118 * x51 + x120 * x170 + x122 * x185 + x140 * x53 +
                                 x147 * x33 + x153 * x41 + x159 * x76 + x165 * x54 + x167 * x61 +
                                 x169 * x68 + x45 * x97;
    element_vector[6 * stride] = x110 * x57 + x118 * x58 + x120 * x175 + x122 * x186 + x140 * x60 +
                                 x148 * x33 + x154 * x41 + x160 * x76 + x167 * x54 + x172 * x61 +
                                 x174 * x68 + x55 * x97;
    element_vector[7 * stride] = x110 * x64 + x118 * x65 + x120 * x178 + x122 * x187 + x140 * x67 +
                                 x149 * x33 + x155 * x41 + x161 * x76 + x169 * x54 + x174 * x61 +
                                 x177 * x68 + x62 * x97;
    element_vector[8 * stride] = x110 * x144 + x118 * x146 + x120 * x179 + x121 * x140 +
                                 x122 * (x10 * x197 + x10 * x198 + x139 + x180 * x85 + x182 * x85 +
                                         x184 + x189 + x193 + x195 + x197 * x21 + x198 * x21) +
                                 x142 * x97 + x185 * x54 + x186 * x61 + x187 * x68 + x191 * x33 +
                                 x194 * x41 + x196 * x76;
    element_vector[9 * stride] = x110 * x71 + x118 * x73 + x120 * x163 + x122 * x196 + x140 * x75 +
                                 x159 * x54 + x160 * x61 + x161 * x68 + x188 * x33 + x192 * x41 +
                                 x195 * x76 + x69 * x97;
    element_vector[10 * stride] = x110 * x27 + x118 * x30 + x120 * x151 + x122 * x191 + x140 * x32 +
                                  x147 * x54 + x148 * x61 + x149 * x68 + x188 * x76 + x189 * x33 +
                                  x190 * x41 + x26 * x97;
    element_vector[11 * stride] = x110 * x37 + x118 * x38 + x120 * x157 + x122 * x194 + x140 * x40 +
                                  x153 * x54 + x154 * x61 + x155 * x68 + x190 * x33 + x192 * x76 +
                                  x193 * x41 + x36 * x97;
}

#endif

void tet4_linear_elasticity_assemble_value_aos(const ptrdiff_t nelements,
                                               const ptrdiff_t nnodes,
                                               idx_t **const SFEM_RESTRICT elems,
                                               geom_t **const SFEM_RESTRICT xyz,
                                               const real_t mu,
                                               const real_t lambda,
                                               const real_t *const SFEM_RESTRICT displacement,
                                               real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int block_size = 3;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];

            real_t element_displacement[(4 * 3)];
#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            for (int enode = 0; enode < 4; ++enode) {
                idx_t dof = ev[enode] * block_size;

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[b * 4 + enode] = displacement[dof + b];
                }
            }

            real_t element_scalar = 0;
            tet4_linear_elasticity_assemble_value_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],

                element_displacement,
                // output vector
                &element_scalar);

#pragma omp atomic update
            *value += element_scalar;
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_linear_elasticity.c: tet4_linear_elasticity_assemble_value_aos\t%g seconds\n",
           tock - tick);
}

void tet4_linear_elasticity_assemble_gradient_aos(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  const real_t mu,
                                                  const real_t lambda,
                                                  const real_t *const SFEM_RESTRICT displacement,
                                                  real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int block_size = 3;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];

            real_t element_vector[(4 * 3)];
            real_t element_displacement[(4 * 3)];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            for (int enode = 0; enode < 4; ++enode) {
                idx_t dof = ev[enode] * block_size;

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[b * 4 + enode] = displacement[dof + b];
                }
            }

            tet4_linear_elasticity_assemble_gradient_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],

                element_displacement,
                // output vector
                element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                for (int b = 0; b < block_size; b++) {
#pragma omp atomic update
                    values[dof_i * block_size + b] += element_vector[b * 4 + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_linear_elasticity.c: tet4_linear_elasticity_assemble_gradient_aos\t%g seconds\n",
           tock - tick);
}

void tet4_linear_elasticity_assemble_hessian_aos(const ptrdiff_t nelements,
                                                 const ptrdiff_t nnodes,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 geom_t **const SFEM_RESTRICT xyz,
                                                 const real_t mu,
                                                 const real_t lambda,
                                                 const count_t *const SFEM_RESTRICT rowptr,
                                                 const idx_t *const SFEM_RESTRICT colidx,
                                                 real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    static const int block_size = 3;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];

            real_t element_matrix[(4 * 3) * (4 * 3)];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            tet4_linear_elasticity_assemble_hessian_kernel(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],

                // output matrix
                element_matrix);

            assert(!check_symmetric(4 * block_size, element_matrix));

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                {
                    const idx_t *row = &colidx[rowptr[dof_i]];
                    find_cols4(ev, row, lenrow, ks);
                }

                // Blocks for row
                real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

                for (int edof_j = 0; edof_j < 4; ++edof_j) {
                    const idx_t offset_j = ks[edof_j] * block_size;

                    for (int bi = 0; bi < block_size; ++bi) {
                        const int ii = bi * 4 + edof_i;

                        // Jump rows (including the block-size for the columns)
                        real_t *row = &block_start[bi * lenrow * block_size];

                        for (int bj = 0; bj < block_size; ++bj) {
                            const int jj = bj * 4 + edof_j;
                            const real_t val = element_matrix[ii * 12 + jj];

#pragma omp atomic update
                            row[offset_j + bj] += val;
                        }
                    }
                }
            }
        }
    }
    const double tock = MPI_Wtime();
    printf("tet4_linear_elasticity.c: tet4_linear_elasticity_assemble_hessian_aos\t%g seconds\n",
           tock - tick);
}

#ifdef SFEM_ENABLE_EXPLICIT_VECTORIZATION

void tet4_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t *const SFEM_RESTRICT displacement,
                                      real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    vreal_t vmu;
    vreal_t vlambda;

    for (int vi = 0; vi < SFEM_VECTOR_SIZE; ++vi) {
        vmu[vi] = mu;
        vlambda[vi] = lambda;
    }

    static const int block_size = 3;

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; i += SFEM_VECTOR_SIZE) {
            const int nvec = MAX(1, MIN(nelements - (i + SFEM_VECTOR_SIZE), SFEM_VECTOR_SIZE));

            idx_t ev[4];
            idx_t ks[4];

            vreal_t x[4];
            vreal_t y[4];
            vreal_t z[4];

            vreal_t element_vector[(4 * 3)];
            vreal_t element_displacement[(4 * 3)];

            for (int vi = 0; vi < nvec; ++vi) {
                const ptrdiff_t offset = i + vi;
#pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    ev[v] = elems[v][offset];
                }

#pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    x[v][vi] = xyz[0][ev[v]];
                    y[v][vi] = xyz[1][ev[v]];
                    z[v][vi] = xyz[2][ev[v]];
                }

                for (int enode = 0; enode < 4; ++enode) {
                    idx_t dof = ev[enode] * block_size;

                    for (int b = 0; b < block_size; ++b) {
                        element_displacement[b * 4 + enode][vi] = displacement[dof + b];
                    }
                }
            }

            tet4_linear_elasticity_apply_kernel(  // Model parameters
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
                element_displacement,
                // output vector
                element_vector);

            for (int vi = 0; vi < nvec; ++vi) {
                const ptrdiff_t offset = i + vi;

#pragma unroll(4)
                for (int v = 0; v < 4; ++v) {
                    ev[v] = elems[v][offset];
                }

                for (int edof_i = 0; edof_i < 4; ++edof_i) {
                    const idx_t dof = ev[edof_i] * block_size;

                    for (int b = 0; b < block_size; b++) {
                        const int evdof_i = b * 4 + edof_i;
#pragma omp atomic update
                        values[dof + b] += element_vector[evdof_i][vi];
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf(
        "tet4_linear_elasticity.c: tet4_linear_elasticity_apply_aos (explicit vectorization)\t%g "
        "seconds\n",
        tock - tick);
}

#else

void tet4_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t *const SFEM_RESTRICT displacement,
                                      real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int block_size = 3;
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];

            real_t element_vector[(4 * 3)];
            real_t element_displacement[(4 * 3)];
#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            for (int enode = 0; enode < 4; ++enode) {
                idx_t dof = ev[enode] * block_size;

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[b * 4 + enode] = displacement[dof + b];
                }
            }

            tet4_linear_elasticity_apply_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],

                element_displacement,
                // output vector
                element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof = ev[edof_i] * block_size;

                for (int b = 0; b < block_size; b++) {
#pragma omp atomic update
                    values[dof + b] += element_vector[b * 4 + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_linear_elasticity.c: tet4_linear_elasticity_apply_aos\t%g seconds\n", tock - tick);
}

#endif

static SFEM_INLINE void tet4_linear_elasticity_apply_kernel_opt(const real_t mu,
                                                                const real_t lambda,
                                                                const real_t px0,
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
                                                                const real_t *const SFEM_RESTRICT u,
                                                                real_t *const SFEM_RESTRICT
                                                                    element_vector) {
    real_t jacobian_inverse[9];
    real_t jacobian_determinant = 0;
    {
        const real_t x0 = -py0 + py2;
        const real_t x1 = -pz0 + pz3;
        const real_t x2 = x0 * x1;
        const real_t x3 = -py0 + py3;
        const real_t x4 = -pz0 + pz2;
        const real_t x5 = x3 * x4;
        const real_t x6 = -px0 + px1;
        const real_t x7 = -pz0 + pz1;
        const real_t x8 = -px0 + px2;
        const real_t x9 = x3 * x8;
        const real_t x10 = -py0 + py1;
        const real_t x11 = -px0 + px3;
        const real_t x12 = x1 * x8;
        const real_t x13 = x0 * x11;
        const real_t x14 = x10 * x11 * x4 - x10 * x12 - x13 * x7 + x2 * x6 - x5 * x6 + x7 * x9;
        const real_t x15 = 1.0 / x14;
        jacobian_inverse[0] = x15 * (x2 - x5);
        jacobian_inverse[1] = x15 * (x11 * x4 - x12);
        jacobian_inverse[2] = x15 * (-x13 + x9);
        jacobian_inverse[3] = x15 * (-x1 * x10 + x3 * x7);
        jacobian_inverse[4] = x15 * (x1 * x6 - x11 * x7);
        jacobian_inverse[5] = x15 * (x10 * x11 - x3 * x6);
        jacobian_inverse[6] = x15 * (-x0 * x7 + x10 * x4);
        jacobian_inverse[7] = x15 * (-x4 * x6 + x7 * x8);
        jacobian_inverse[8] = x15 * (x0 * x6 - x10 * x8);
        jacobian_determinant = x14;
    }

    real_t buff[9];
    {
        const real_t x0 = -jacobian_inverse[0] - jacobian_inverse[3] - jacobian_inverse[6];
        const real_t x1 = -jacobian_inverse[1] - jacobian_inverse[4] - jacobian_inverse[7];
        const real_t x2 = -jacobian_inverse[2] - jacobian_inverse[5] - jacobian_inverse[8];
        buff[0] = jacobian_inverse[0] * u[1] + jacobian_inverse[3] * u[2] +
                       jacobian_inverse[6] * u[3] + u[0] * x0;
        buff[1] = jacobian_inverse[1] * u[1] + jacobian_inverse[4] * u[2] +
                       jacobian_inverse[7] * u[3] + u[0] * x1;
        buff[2] = jacobian_inverse[2] * u[1] + jacobian_inverse[5] * u[2] +
                       jacobian_inverse[8] * u[3] + u[0] * x2;
        buff[3] = jacobian_inverse[0] * u[5] + jacobian_inverse[3] * u[6] +
                       jacobian_inverse[6] * u[7] + u[4] * x0;
        buff[4] = jacobian_inverse[1] * u[5] + jacobian_inverse[4] * u[6] +
                       jacobian_inverse[7] * u[7] + u[4] * x1;
        buff[5] = jacobian_inverse[2] * u[5] + jacobian_inverse[5] * u[6] +
                       jacobian_inverse[8] * u[7] + u[4] * x2;
        buff[6] = jacobian_inverse[0] * u[9] + jacobian_inverse[3] * u[10] +
                       jacobian_inverse[6] * u[11] + u[8] * x0;
        buff[7] = jacobian_inverse[1] * u[9] + jacobian_inverse[4] * u[10] +
                       jacobian_inverse[7] * u[11] + u[8] * x1;
        buff[8] = jacobian_inverse[2] * u[9] + jacobian_inverse[5] * u[10] +
                       jacobian_inverse[8] * u[11] + u[8] * x2;
    }

    real_t P[9];
    {
        const real_t x0 = 2 * buff[0];
        const real_t x1 = 2 * buff[4];
        const real_t x2 = 2 * buff[8];
        const real_t x3 = (1.0 / 2.0) * lambda * (x0 + x1 + x2);
        const real_t x4 = mu * (buff[1] + buff[3]);
        const real_t x5 = mu * (buff[2] + buff[6]);
        const real_t x6 = mu * (buff[5] + buff[7]);
        P[0] = mu * x0 + x3;
        P[1] = x4;
        P[2] = x5;
        P[3] = x4;
        P[4] = mu * x1 + x3;
        P[5] = x6;
        P[6] = x5;
        P[7] = x6;
        P[8] = mu * x2 + x3;
    }

    // buff = det(J)J^-1 * P
    {
        buff[0] =
            jacobian_determinant *
            (P[0] * jacobian_inverse[0] + P[3] * jacobian_inverse[1] + P[6] * jacobian_inverse[2]);
        buff[1] =
            jacobian_determinant *
            (P[1] * jacobian_inverse[0] + P[4] * jacobian_inverse[1] + P[7] * jacobian_inverse[2]);
        buff[2] =
            jacobian_determinant *
            (P[2] * jacobian_inverse[0] + P[5] * jacobian_inverse[1] + P[8] * jacobian_inverse[2]);
        buff[3] =
            jacobian_determinant *
            (P[0] * jacobian_inverse[3] + P[3] * jacobian_inverse[4] + P[6] * jacobian_inverse[5]);
        buff[4] =
            jacobian_determinant *
            (P[1] * jacobian_inverse[3] + P[4] * jacobian_inverse[4] + P[7] * jacobian_inverse[5]);
        buff[5] =
            jacobian_determinant *
            (P[2] * jacobian_inverse[3] + P[5] * jacobian_inverse[4] + P[8] * jacobian_inverse[5]);
        buff[6] =
            jacobian_determinant *
            (P[0] * jacobian_inverse[6] + P[3] * jacobian_inverse[7] + P[6] * jacobian_inverse[8]);
        buff[7] =
            jacobian_determinant *
            (P[1] * jacobian_inverse[6] + P[4] * jacobian_inverse[7] + P[7] * jacobian_inverse[8]);
        buff[8] =
            jacobian_determinant *
            (P[2] * jacobian_inverse[6] + P[5] * jacobian_inverse[7] + P[8] * jacobian_inverse[8]);
    }

    // Evaluate bilinear form
    {
        const real_t x0 = (1.0 / 6.0) * buff[0];
        const real_t x1 = (1.0 / 6.0) * buff[1];
        const real_t x2 = (1.0 / 6.0) * buff[2];
        const real_t x3 = (1.0 / 6.0) * buff[3];
        const real_t x4 = (1.0 / 6.0) * buff[4];
        const real_t x5 = (1.0 / 6.0) * buff[5];
        const real_t x6 = (1.0 / 6.0) * buff[6];
        const real_t x7 = (1.0 / 6.0) * buff[7];
        const real_t x8 = (1.0 / 6.0) * buff[8];
        element_vector[0 * stride] = -x0 - x1 - x2;
        element_vector[1 * stride] = x0;
        element_vector[2 * stride] = x1;
        element_vector[3 * stride] = x2;
        element_vector[4 * stride] = -x3 - x4 - x5;
        element_vector[5 * stride] = x3;
        element_vector[6 * stride] = x4;
        element_vector[7 * stride] = x5;
        element_vector[8 * stride] = -x6 - x7 - x8;
        element_vector[9 * stride] = x6;
        element_vector[10 * stride] = x7;
        element_vector[11 * stride] = x8;
    }
}

void tet4_linear_elasticity_apply_soa(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t **const SFEM_RESTRICT u,
                                      real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

    idx_t ev[4];
    real_t element_vector[(4 * 3)];
    real_t element_displacement[(4 * 3)];

    static const int block_size = 3;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int b = 0; b < block_size; b++) {
            for (int v = 0; v < 4; ++v) {
                element_displacement[b * 4 + v] = u[b][ev[v]];
            }
        }

        tet4_linear_elasticity_apply_kernel_opt(
            // Model parameters
            mu,
            lambda,
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
            // input
            element_displacement,
            // output
            element_vector);

        for (int bi = 0; bi < block_size; ++bi) {
            for (int edof_i = 0; edof_i < 4; edof_i++) {
                values[bi][ev[edof_i]] += element_vector[bi * 4 + edof_i];
            }
        }
    }
}
