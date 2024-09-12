// #include "laplacian.h"

#include <cassert>
#include <cmath>
// #include <cstdio>
#include <algorithm>
#include <cstddef>

// #include <mpi.h>

extern "C" {
#include "sfem_base.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"
}

#include "sfem_cuda_base.h"

#define POW2(a) ((a) * (a))

// Version 1
// static inline __device__ void laplacian(const real_t x0,
//                                         const real_t x1,
//                                         const real_t x2,
//                                         const real_t x3,
//                                         const real_t y0,
//                                         const real_t y1,
//                                         const real_t y2,
//                                         const real_t y3,
//                                         const real_t z0,
//                                         const real_t z1,
//                                         const real_t z2,
//                                         const real_t z3,
//                                         real_t *element_matrix) {
//     // FLOATING POINT OPS!
//     //    - Result: 4*ADD + 16*ASSIGNMENT + 16*MUL + 12*POW
//     //    - Subexpressions: 16*ADD + 9*DIV + 56*MUL + 7*NEG + POW + 32*SUB
//     const real_t x4 = z0 - z3;
//     const real_t x5 = x0 - x1;
//     const real_t x6 = y0 - y2;
//     const real_t x7 = x5 * x6;
//     const real_t x8 = z0 - z1;
//     const real_t x9 = x0 - x2;
//     const real_t x10 = y0 - y3;
//     const real_t x11 = x10 * x9;
//     const real_t x12 = z0 - z2;
//     const real_t x13 = x0 - x3;
//     const real_t x14 = y0 - y1;
//     const real_t x15 = x13 * x14;
//     const real_t x16 = x10 * x5;
//     const real_t x17 = x14 * x9;
//     const real_t x18 = x13 * x6;
//     const real_t x19 = x11 * x8 + x12 * x15 - x12 * x16 - x17 * x4 - x18 * x8 + x4 * x7;
//     const real_t x20 = 1.0 / x19;
//     const real_t x21 = x11 - x18;
//     const real_t x22 = -x17 + x7;
//     const real_t x23 = x15 - x16 + x21 + x22;
//     const real_t x24 = -x12 * x13 + x4 * x9;
//     const real_t x25 = x12 * x5 - x8 * x9;
//     const real_t x26 = x13 * x8;
//     const real_t x27 = x4 * x5;
//     const real_t x28 = x26 - x27;
//     const real_t x29 = -x24 - x25 - x28;
//     const real_t x30 = x10 * x8;
//     const real_t x31 = x14 * x4;
//     const real_t x32 = -x10 * x12 + x4 * x6;
//     const real_t x33 = x12 * x14 - x6 * x8;
//     const real_t x34 = x30 - x31 + x32 + x33;
//     const real_t x35 = -x12;
//     const real_t x36 = -x9;
//     const real_t x37 = x19 * (x13 * x35 + x28 - x35 * x5 - x36 * x4 + x36 * x8);
//     const real_t x38 = -x19;
//     const real_t x39 = -x23;
//     const real_t x40 = -x34;
//     const real_t x41 = (1.0 / 6.0) / pow(x19, 2);
//     const real_t x42 = x41 * (x24 * x37 + x38 * (x21 * x39 + x32 * x40));
//     const real_t x43 = -x15 + x16;
//     const real_t x44 = (1.0 / 6.0) * x43;
//     const real_t x45 = -x26 + x27;
//     const real_t x46 = -x30 + x31;
//     const real_t x47 = (1.0 / 6.0) * x46;
//     const real_t x48 = x20 * (-x23 * x44 + (1.0 / 6.0) * x29 * x45 - x34 * x47);
//     const real_t x49 = x41 * (x25 * x37 + x38 * (x22 * x39 + x33 * x40));
//     const real_t x50 = (1.0 / 6.0) * x45;
//     const real_t x51 = x20 * (x21 * x44 + x24 * x50 + x32 * x47);
//     const real_t x52 = x20 * (-1.0 / 6.0 * x21 * x22 - 1.0 / 6.0 * x24 * x25 - 1.0 / 6.0 * x32 * x33);
//     const real_t x53 = x20 * (x22 * x44 + x25 * x50 + x33 * x47);

//     element_matrix[0] = x20 * (-1.0 / 6.0 * pow(x23, 2) - 1.0 / 6.0 * pow(x29, 2) - 1.0 / 6.0 * pow(x34, 2));
//     element_matrix[1] = x42;
//     element_matrix[2] = x48;
//     element_matrix[3] = x49;
//     element_matrix[4] = x42;
//     element_matrix[5] = x20 * (-1.0 / 6.0 * pow(x21, 2) - 1.0 / 6.0 * pow(x24, 2) - 1.0 / 6.0 * pow(x32, 2));
//     element_matrix[6] = x51;
//     element_matrix[7] = x52;
//     element_matrix[8] = x48;
//     element_matrix[9] = x51;
//     element_matrix[10] = x20 * (-1.0 / 6.0 * pow(x43, 2) - 1.0 / 6.0 * pow(x45, 2) - 1.0 / 6.0 * pow(x46, 2));
//     element_matrix[11] = x53;
//     element_matrix[12] = x49;
//     element_matrix[13] = x52;
//     element_matrix[14] = x53;
//     element_matrix[15] = x20 * (-1.0 / 6.0 * pow(x22, 2) - 1.0 / 6.0 * pow(x25, 2) - 1.0 / 6.0 * pow(x33, 2));
// }

// Version 2
// static inline __device__ void laplacian(const real_t px0,
//                                         const real_t px1,
//                                         const real_t px2,
//                                         const real_t px3,
//                                         const real_t py0,
//                                         const real_t py1,
//                                         const real_t py2,
//                                         const real_t py3,
//                                         const real_t pz0,
//                                         const real_t pz1,
//                                         const real_t pz2,
//                                         const real_t pz3,
//                                         real_t *element_matrix)

// {
//     real_t jac_inv[9];
//     real_t dv;
//     {
//         // FLOATING POINT OPS!
//         //      - Result: 10*ADD + 10*ASSIGNMENT + 31*MUL
//         //      - Subexpressions: 2*ADD + DIV + 12*MUL + 12*SUB
//         const real_t x0 = -py0 + py2;
//         const real_t x1 = -pz0 + pz3;
//         const real_t x2 = x0 * x1;
//         const real_t x3 = -py0 + py3;
//         const real_t x4 = -pz0 + pz2;
//         const real_t x5 = x3 * x4;
//         const real_t x6 = -px0 + px1;
//         const real_t x7 = x2 * x6;
//         const real_t x8 = -pz0 + pz1;
//         const real_t x9 = -px0 + px2;
//         const real_t x10 = x3 * x9;
//         const real_t x11 = x10 * x8;
//         const real_t x12 = -py0 + py1;
//         const real_t x13 = -px0 + px3;
//         const real_t x14 = x12 * x13 * x4;
//         const real_t x15 = x5 * x6;
//         const real_t x16 = x1 * x9;
//         const real_t x17 = x12 * x16;
//         const real_t x18 = x0 * x13;
//         const real_t x19 = x18 * x8;
//         const real_t x20 = 1.0 / (x11 + x14 - x15 - x17 - x19 + x7);
//         jac_inv[0] = x20 * (x2 - x5);
//         jac_inv[1] = x20 * (x13 * x4 - x16);
//         jac_inv[2] = x20 * (x10 - x18);
//         jac_inv[3] = x20 * (-x1 * x12 + x3 * x8);
//         jac_inv[4] = x20 * (x1 * x6 - x13 * x8);
//         jac_inv[5] = x20 * (x12 * x13 - x3 * x6);
//         jac_inv[6] = x20 * (-x0 * x8 + x12 * x4);
//         jac_inv[7] = x20 * (-x4 * x6 + x8 * x9);
//         jac_inv[8] = x20 * (x0 * x6 - x12 * x9);
//         dv = (1.0 / 6.0) * x11 + (1.0 / 6.0) * x14 - 1.0 / 6.0 * x15 - 1.0 / 6.0 * x17 - 1.0 / 6.0 * x19 +
//              (1.0 / 6.0) * x7;
//     }

//     {
//         // FLOATING POINT OPS!
//         //      - Result: 10*ADD + 20*ASSIGNMENT + 30*MUL + 12*POW
//         //      - Subexpressions: 2*ADD + 6*DIV + 18*MUL + 3*NEG + 18*SUB
//         const real_t x0 = -jac_inv[0] - jac_inv[3] - jac_inv[6];
//         const real_t x1 = -pz0 + pz3;
//         const real_t x2 = -py0 + py2;
//         const real_t x3 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px1;
//         const real_t x4 = -py0 + py3;
//         const real_t x5 = -pz0 + pz2;
//         const real_t x6 = -py0 + py1;
//         const real_t x7 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px2;
//         const real_t x8 = -pz0 + pz1;
//         const real_t x9 = -1.0 / 6.0 * px0 + (1.0 / 6.0) * px3;
//         const real_t x10 = x1 * x2 * x3 - x1 * x6 * x7 - x2 * x8 * x9 - x3 * x4 * x5 + x4 * x7 * x8 + x5 * x6 * x9;
//         const real_t x11 = -jac_inv[1] - jac_inv[4] - jac_inv[7];
//         const real_t x12 = -jac_inv[2] - jac_inv[5] - jac_inv[8];
//         const real_t x13 = x0 * x10;
//         const real_t x14 = x10 * x11;
//         const real_t x15 = x10 * x12;
//         const real_t x16 = jac_inv[0] * x10;
//         const real_t x17 = jac_inv[1] * x10;
//         const real_t x18 = jac_inv[2] * x10;
//         element_matrix[0] = pow(x0, 2) * x10 + x10 * pow(x11, 2) + x10 * pow(x12, 2);
//         element_matrix[0] = element_matrix[0];
//         element_matrix[1] = jac_inv[0] * x13 + jac_inv[1] * x14 + jac_inv[2] * x15;
//         element_matrix[4] = element_matrix[1];
//         element_matrix[2] = jac_inv[3] * x13 + jac_inv[4] * x14 + jac_inv[5] * x15;
//         element_matrix[8] = element_matrix[2];
//         element_matrix[3] = jac_inv[6] * x13 + jac_inv[7] * x14 + jac_inv[8] * x15;
//         element_matrix[12] = element_matrix[3];
//         element_matrix[5] = pow(jac_inv[0], 2) * x10 + pow(jac_inv[1], 2) * x10 + pow(jac_inv[2], 2) * x10;
//         element_matrix[5] = element_matrix[5];
//         element_matrix[6] = jac_inv[3] * x16 + jac_inv[4] * x17 + jac_inv[5] * x18;
//         element_matrix[9] = element_matrix[6];
//         element_matrix[7] = jac_inv[6] * x16 + jac_inv[7] * x17 + jac_inv[8] * x18;
//         element_matrix[13] = element_matrix[7];
//         element_matrix[10] = pow(jac_inv[3], 2) * x10 + pow(jac_inv[4], 2) * x10 + pow(jac_inv[5], 2) * x10;
//         element_matrix[10] = element_matrix[10];
//         element_matrix[11] =
//             jac_inv[3] * jac_inv[6] * x10 + jac_inv[4] * jac_inv[7] * x10 + jac_inv[5] * jac_inv[8] * x10;
//         element_matrix[14] = element_matrix[11];
//         element_matrix[15] = pow(jac_inv[6], 2) * x10 + pow(jac_inv[7], 2) * x10 + pow(jac_inv[8], 2) * x10;
//     }
// }

// Version 3
static inline __device__ void laplacian(const real_t px0,
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
                                        real_t *element_matrix) {
    {
        real_t jac_inv[9];
        real_t dv;
        {
            // FLOATING POINT OPS!
            //       - Result: 10*ADD + 10*ASSIGNMENT + 31*MUL
            //       - Subexpressions: 2*ADD + DIV + 12*MUL + 12*SUB
            const real_t x0 = -py0 + py2;
            const real_t x1 = -pz0 + pz3;
            const real_t x2 = x0 * x1;
            const real_t x3 = -py0 + py3;
            const real_t x4 = -pz0 + pz2;
            const real_t x5 = x3 * x4;
            const real_t x6 = -px0 + px1;
            const real_t x7 = x2 * x6;
            const real_t x8 = -pz0 + pz1;
            const real_t x9 = -px0 + px2;
            const real_t x10 = x3 * x9;
            const real_t x11 = x10 * x8;
            const real_t x12 = -py0 + py1;
            const real_t x13 = -px0 + px3;
            const real_t x14 = x12 * x13 * x4;
            const real_t x15 = x5 * x6;
            const real_t x16 = x1 * x9;
            const real_t x17 = x12 * x16;
            const real_t x18 = x0 * x13;
            const real_t x19 = x18 * x8;
            const real_t x20 = 1.0 / (x11 + x14 - x15 - x17 - x19 + x7);
            jac_inv[0] = x20 * (x2 - x5);
            jac_inv[1] = x20 * (x13 * x4 - x16);
            jac_inv[2] = x20 * (x10 - x18);
            jac_inv[3] = x20 * (-x1 * x12 + x3 * x8);
            jac_inv[4] = x20 * (x1 * x6 - x13 * x8);
            jac_inv[5] = x20 * (x12 * x13 - x3 * x6);
            jac_inv[6] = x20 * (-x0 * x8 + x12 * x4);
            jac_inv[7] = x20 * (-x4 * x6 + x8 * x9);
            jac_inv[8] = x20 * (x0 * x6 - x12 * x9);
            dv = (1.0 / 6.0) * x11 + (1.0 / 6.0) * x14 - 1.0 / 6.0 * x15 - 1.0 / 6.0 * x17 - 1.0 / 6.0 * x19 +
                 (1.0 / 6.0) * x7;
        }

        real_t gx[4], gy[4], gz[4];
        {
            // FLOATING POINT OPS!
            //       - Result: 3*ADD + 12*ASSIGNMENT + 9*MUL

            gx[0] = -jac_inv[0] - jac_inv[3] - jac_inv[6];
            gy[0] = -jac_inv[1] - jac_inv[4] - jac_inv[7];
            gz[0] = -jac_inv[2] - jac_inv[5] - jac_inv[8];
            gx[1] = jac_inv[0];
            gy[1] = jac_inv[1];
            gz[1] = jac_inv[2];
            gx[2] = jac_inv[3];
            gy[2] = jac_inv[4];
            gz[2] = jac_inv[5];
            gx[3] = jac_inv[6];
            gy[3] = jac_inv[7];
            gz[3] = jac_inv[8];
        }

        {
            // FLOATING POINT OPS!
            //      - Result: 10*ADD + 20*ASSIGNMENT + 28*MUL + 12*POW
            //      - Subexpressions: 0
            element_matrix[0] = dv * (POW2(gx[0]) + POW2(gy[0]) + POW2(gz[0]));
            element_matrix[0] = element_matrix[0];
            element_matrix[1] = dv * (gx[0] * gx[1] + gy[0] * gy[1] + gz[0] * gz[1]);
            element_matrix[4] = element_matrix[1];
            element_matrix[2] = dv * (gx[0] * gx[2] + gy[0] * gy[2] + gz[0] * gz[2]);
            element_matrix[8] = element_matrix[2];
            element_matrix[3] = dv * (gx[0] * gx[3] + gy[0] * gy[3] + gz[0] * gz[3]);
            element_matrix[12] = element_matrix[3];
            element_matrix[5] = dv * (POW2(gx[1]) + POW2(gy[1]) + POW2(gz[1]));
            element_matrix[5] = element_matrix[5];
            element_matrix[6] = dv * (gx[1] * gx[2] + gy[1] * gy[2] + gz[1] * gz[2]);
            element_matrix[9] = element_matrix[6];
            element_matrix[7] = dv * (gx[1] * gx[3] + gy[1] * gy[3] + gz[1] * gz[3]);
            element_matrix[13] = element_matrix[7];
            element_matrix[10] = dv * (POW2(gx[2]) + POW2(gy[2]) + POW2(gz[2]));
            element_matrix[10] = element_matrix[10];
            element_matrix[11] = dv * (gx[2] * gx[3] + gy[2] * gy[3] + gz[2] * gz[3]);
            element_matrix[14] = element_matrix[11];
            element_matrix[15] = dv * (POW2(gx[3]) + POW2(gy[3]) + POW2(gz[3]));
            element_matrix[15] = element_matrix[15];
        }
    }
}

static inline __device__ void laplacian_gradient(const real_t px0,
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
                                                 const real_t *SFEM_RESTRICT u,
                                                 real_t *SFEM_RESTRICT element_vector) {
    // FLOATING POINT OPS!
    //      - Result: 4*ADD + 4*ASSIGNMENT + 16*MUL
    //      - Subexpressions: 13*ADD + 7*DIV + 46*MUL + 3*NEG + 30*SUB
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py2;
    const real_t x3 = x1 * x2;
    const real_t x4 = x0 * x3;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 = -py0 + py3;
    const real_t x7 = x1 * x6;
    const real_t x8 = x5 * x7;
    const real_t x9 = -px0 + px2;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x9;
    const real_t x12 = x0 * x11;
    const real_t x13 = -pz0 + pz1;
    const real_t x14 = x6 * x9;
    const real_t x15 = x13 * x14;
    const real_t x16 = -px0 + px3;
    const real_t x17 = x10 * x16 * x5;
    const real_t x18 = x16 * x2;
    const real_t x19 = x13 * x18;
    const real_t x20 =
        -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 + (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
    const real_t x21 = 1.0 / (-x12 + x15 + x17 - x19 + x4 - x8);
    const real_t x22 = x21 * (-x11 + x3);
    const real_t x23 = x21 * (x10 * x16 - x7);
    const real_t x24 = x21 * (x14 - x18);
    const real_t x25 = -x22 - x23 - x24;
    const real_t x26 = u[0] * x25 + u[1] * x24 + u[2] * x23 + u[3] * x22;
    const real_t x27 = x21 * (-x1 * x5 + x13 * x9);
    const real_t x28 = x21 * (x0 * x1 - x13 * x16);
    const real_t x29 = x21 * (-x0 * x9 + x16 * x5);
    const real_t x30 = -x27 - x28 - x29;
    const real_t x31 = u[0] * x30 + u[1] * x29 + u[2] * x28 + u[3] * x27;
    const real_t x32 = x21 * (x10 * x5 - x13 * x2);
    const real_t x33 = x21 * (-x0 * x10 + x13 * x6);
    const real_t x34 = x21 * (x0 * x2 - x5 * x6);
    const real_t x35 = -x32 - x33 - x34;
    const real_t x36 = u[0] * x35 + u[1] * x34 + u[2] * x33 + u[3] * x32;
    element_vector[0] = x20 * (x25 * x26 + x30 * x31 + x35 * x36);
    element_vector[1] = x20 * (x24 * x26 + x29 * x31 + x34 * x36);
    element_vector[2] = x20 * (x23 * x26 + x28 * x31 + x33 * x36);
    element_vector[3] = x20 * (x22 * x26 + x27 * x31 + x32 * x36);
}

static inline __device__ int linear_search(const idx_t target, const idx_t *const arr, const int size) {
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

static inline __device__ int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    // if (lenrow <= 32)
    // {
    return linear_search(key, row, lenrow);

    // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
    // while (key > row[++k]) {
    //     // Hi
    // }
    // assert(k < lenrow);
    // assert(key == row[k]);
    // } else {
    //     // Use this for larger number of dofs per row
    //     return find_idx_binary_search(key, row, lenrow);
    // }
}

static inline __device__ void find_cols4(const idx_t *targets, const idx_t *const row, const int lenrow, int *ks) {
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

__global__ void laplacian_crs_kernel(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  count_t *const SFEM_RESTRICT rowptr,
                                                  idx_t *const SFEM_RESTRICT colidx,
                                                  real_t *const SFEM_RESTRICT values) {
    idx_t ev[4];
    idx_t ks[4];
    real_t element_matrix[4 * 4];

    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements; i += blockDim.x * gridDim.x) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        laplacian(
            // X-coordinates
            xyz[0][ev[0]],
            xyz[0][ev[1]],
            xyz[0][ev[2]],
            xyz[0][ev[3]],
            // Y-coordinates
            xyz[1][ev[0]],
            xyz[1][ev[1]],
            xyz[1][ev[2]],
            xyz[1][ev[3]],
            // Z-coordinates
            xyz[2][ev[0]],
            xyz[2][ev[1]],
            xyz[2][ev[2]],
            xyz[2][ev[3]],
            element_matrix);

#pragma unroll(4)
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            find_cols4(ev, row, lenrow, ks);

            real_t *rowvalues = &values[rowptr[dof_i]];
            const real_t *element_row = &element_matrix[edof_i * 4];

#pragma unroll(4)
            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                real_t v = element_row[edof_j];
                atomicAdd(&rowvalues[ks[edof_j]], v);
            }
        }
    }
}

__global__ void print_elem_kernel(const ptrdiff_t nelements, idx_t **const elems) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nelements) return;

    printf("%d %d %d %d\n", elems[0][i], elems[1][i], elems[2][i], elems[3][i]);
}

extern "C" void tet4_laplacian_crs(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elems,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values) {
    // double tick = MPI_Wtime();

    const ptrdiff_t nbatch = nelements;

    idx_t *hd_elems[4];
    idx_t **d_elems = nullptr;

    {  // Copy element indices

        void *ptr;
        SFEM_CUDA_CHECK(cudaMalloc(&ptr, 4 * sizeof(idx_t *)));
        d_elems = (idx_t **)ptr;

        for (int d = 0; d < 4; ++d) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(hd_elems[d], elems[d], nbatch * sizeof(idx_t), cudaMemcpyHostToDevice));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(d_elems, hd_elems, 4 * sizeof(idx_t *), cudaMemcpyHostToDevice));
    }

    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

    geom_t *hd_xyz[4];
    geom_t **d_xyz = nullptr;

    {  // Copy coordinates
        SFEM_CUDA_CHECK(cudaMalloc(&d_xyz, 3 * sizeof(geom_t *)));

        for (int d = 0; d < 3; ++d) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_xyz[d], nnodes * sizeof(geom_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(hd_xyz[d], xyz[d], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(d_xyz, hd_xyz, 3 * sizeof(geom_t *), cudaMemcpyHostToDevice));
    }

    count_t *d_rowptr = nullptr;
    idx_t *d_colidx = nullptr;
    real_t *d_values = nullptr;
    const count_t nnz = rowptr[nnodes];

    {  // Copy matrix
        SFEM_CUDA_CHECK(cudaMalloc(&d_rowptr, (nnodes + 1) * sizeof(count_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(d_rowptr, rowptr, (nnodes + 1) * sizeof(count_t), cudaMemcpyHostToDevice));

        SFEM_CUDA_CHECK(cudaMalloc(&d_colidx, nnz * sizeof(idx_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(d_colidx, colidx, nnz * sizeof(idx_t), cudaMemcpyHostToDevice));

        SFEM_CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(real_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(d_values, values, nnz * sizeof(real_t), cudaMemcpyHostToDevice));
    }

    // double ktick = MPI_Wtime();
    {
        laplacian_crs_kernel<<<n_blocks, block_size>>>(
            nelements, nnodes, d_elems, d_xyz, d_rowptr, d_colidx, d_values);
        SFEM_DEBUG_SYNCHRONIZE();

        // cudaDeviceSynchronize();
        // double ktock = MPI_Wtime();
        // printf("cuda_laplacian.c: laplacian_crs_kernel\t%g seconds\n", ktock - ktick);

        // Copy result to Host memory
        SFEM_CUDA_CHECK(cudaMemcpy(values, d_values, nnz * sizeof(real_t), cudaMemcpyDeviceToHost));
    }
    // double ktock = MPI_Wtime();

    {  // Free element indices
        for (int d = 0; d < 4; ++d) {
            SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
        }

        SFEM_CUDA_CHECK(cudaFree(d_elems));
    }

    {  // Free element coordinates
        for (int d = 0; d < 3; ++d) {
            SFEM_CUDA_CHECK(cudaFree(hd_xyz[d]));
        }

        SFEM_CUDA_CHECK(cudaFree(d_xyz));
    }

    {  // Free matrix
        SFEM_CUDA_CHECK(cudaFree(d_rowptr));
        SFEM_CUDA_CHECK(cudaFree(d_colidx));
        SFEM_CUDA_CHECK(cudaFree(d_values));
    }

    // double tock = MPI_Wtime();
    // printf("cuda_laplacian.c: laplacian_crs\t%g seconds (GPU kernel %g seconds)\n",
    //        tock - tick,
    //        ktock - ktick);
}

__global__ void laplacian_assemble_gradient_kernel(const ptrdiff_t nelements,
                                                   const ptrdiff_t nnodes,
                                                   idx_t **const SFEM_RESTRICT elems,
                                                   geom_t **const SFEM_RESTRICT xyz,
                                                   const real_t *const SFEM_RESTRICT u,
                                                   real_t *const SFEM_RESTRICT values) {
    idx_t ev[4];
    real_t element_vector[4];
    real_t element_u[4];

    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements; i += blockDim.x * gridDim.x) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_u[v] = u[ev[v]];
        }

        laplacian_gradient(
            // X-coordinates
            xyz[0][ev[0]],
            xyz[0][ev[1]],
            xyz[0][ev[2]],
            xyz[0][ev[3]],
            // Y-coordinates
            xyz[1][ev[0]],
            xyz[1][ev[1]],
            xyz[1][ev[2]],
            xyz[1][ev[3]],
            // Z-coordinates
            xyz[2][ev[0]],
            xyz[2][ev[1]],
            xyz[2][ev[2]],
            xyz[2][ev[3]],
            element_u,
            element_vector);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            atomicAdd(&values[dof_i], element_vector[edof_i]);
        }
    }
}

extern "C" void tet4_laplacian_assemble_gradient(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t *const SFEM_RESTRICT u,
                                            real_t *const SFEM_RESTRICT values) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    SFEM_RANGE_PUSH("lapl-set-up");
    cudaEventRecord(start);

    const ptrdiff_t nbatch = nelements;

    idx_t *hd_elems[4];
    idx_t **d_elems = nullptr;

    {  // Copy element indices

        void *ptr;
        SFEM_CUDA_CHECK(cudaMalloc(&ptr, 4 * sizeof(idx_t *)));
        d_elems = (idx_t **)ptr;

        for (int d = 0; d < 4; ++d) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(hd_elems[d], elems[d], nbatch * sizeof(idx_t), cudaMemcpyHostToDevice));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(d_elems, hd_elems, 4 * sizeof(idx_t *), cudaMemcpyHostToDevice));
    }

    static int block_size = 128;
    ptrdiff_t n_blocks = std::max(ptrdiff_t(1), (nbatch + block_size - 1) / block_size);

    geom_t *hd_xyz[4];
    geom_t **d_xyz = nullptr;

    {  // Copy coordinates
        SFEM_CUDA_CHECK(cudaMalloc(&d_xyz, 3 * sizeof(geom_t *)));

        for (int d = 0; d < 3; ++d) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_xyz[d], nnodes * sizeof(geom_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(hd_xyz[d], xyz[d], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(d_xyz, hd_xyz, 3 * sizeof(geom_t *), cudaMemcpyHostToDevice));
    }

    real_t *d_u = nullptr;
    real_t *d_values = nullptr;
    {
        // Copy input and output to device
        SFEM_CUDA_CHECK(cudaMalloc(&d_u, nnodes * sizeof(real_t)));
        SFEM_CUDA_CHECK(cudaMalloc(&d_values, nnodes * sizeof(real_t)));

        SFEM_CUDA_CHECK(cudaMemcpy(d_u, u, nnodes * sizeof(real_t), cudaMemcpyHostToDevice));
        SFEM_CUDA_CHECK(cudaMemcpy(d_values, values, nnodes * sizeof(real_t), cudaMemcpyHostToDevice));
    }

    SFEM_RANGE_POP();

    // double ktick = MPI_Wtime();
    laplacian_assemble_gradient_kernel<<<n_blocks, block_size>>>(nelements, nnodes, d_elems, d_xyz, d_u, d_values);

    SFEM_CUDA_CHECK(cudaMemcpy(values, d_values, nnodes * sizeof(real_t), cudaMemcpyDeviceToHost));
    // double ktock = MPI_Wtime();

    SFEM_RANGE_PUSH("lapl-tear-down");
    {  // Free element indices
        for (int d = 0; d < 4; ++d) {
            SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
        }

        SFEM_CUDA_CHECK(cudaFree(d_elems));
    }

    {  // Free element coordinates
        for (int d = 0; d < 3; ++d) {
            SFEM_CUDA_CHECK(cudaFree(hd_xyz[d]));
        }

        SFEM_CUDA_CHECK(cudaFree(d_xyz));
    }

    {
        SFEM_CUDA_CHECK(cudaFree(d_u));
        SFEM_CUDA_CHECK(cudaFree(d_values));
    }

    SFEM_RANGE_POP();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // printf("cuda_laplacian.c: laplacian_assemble_gradient\t%g seconds\nloops %d\n",
    //        milliseconds / 1000,
    //        int(nelements / nbatch));
}

extern "C" void tet4_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values) {
   tet4_laplacian_assemble_gradient(nelements, nnodes, elems, xyz, u, values);
}

extern "C" void tet4_laplacian_assemble_value(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elems,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT value)
                              {
                                assert(false);
                              }
