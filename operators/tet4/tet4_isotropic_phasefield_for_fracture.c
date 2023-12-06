#include "isotropic_phasefield_for_fracture.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE real_t tet4_measure(
    // x
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // z
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
    return (x0 * x1 * x2 - x0 * x4 * x8 - x1 * x5 * x6 - x2 * x3 * x7 + x3 * x4 * x5 + x6 * x7 * x8) / 6;
}

//
static SFEM_INLINE void tet4_fe_tgrad(
    // x
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // z
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3,
    // Coefficients
    const real_t *c,

    // Out
    real_t *output) {
    // FLOATING POINT OPS!
    //  - Result: 9*ADD + 9*ASSIGNMENT + 36*MUL
    //  - Subexpressions: 6*ADD + DIV + 36*MUL + 4*NEG + 26*SUB
    const real_t x0 = py0 - py2;
    const real_t x1 = pz0 - pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = py0 - py3;
    const real_t x4 = pz0 - pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = -x6;
    const real_t x8 = px0 - px1;
    const real_t x9 = px0 - px2;
    const real_t x10 = pz0 - pz1;
    const real_t x11 = x10 * x3;
    const real_t x12 = px0 - px3;
    const real_t x13 = py0 - py1;
    const real_t x14 = x13 * x4;
    const real_t x15 = x1 * x13;
    const real_t x16 = x0 * x10;
    const real_t x17 = 1.0 / (x11 * x9 + x12 * x14 - x12 * x16 - x15 * x9 + x2 * x8 - x5 * x8);
    const real_t x18 = c[3] * x17;
    const real_t x19 = -x11 + x15;
    const real_t x20 = c[6] * x17;
    const real_t x21 = x14 - x16;
    const real_t x22 = -x21;
    const real_t x23 = c[9] * x17;
    const real_t x24 = x11 - x15 + x21 + x6;
    const real_t x25 = c[0] * x17;
    const real_t x26 = x1 * x9 - x12 * x4;
    const real_t x27 = x1 * x8;
    const real_t x28 = x10 * x12;
    const real_t x29 = -x27 + x28;
    const real_t x30 = -x10 * x9 + x4 * x8;
    const real_t x31 = -x26 + x27 - x28 - x30;
    const real_t x32 = -x0 * x12 + x3 * x9;
    const real_t x33 = -x32;
    const real_t x34 = x3 * x8;
    const real_t x35 = x12 * x13;
    const real_t x36 = x34 - x35;
    const real_t x37 = x0 * x8 - x13 * x9;
    const real_t x38 = -x37;
    const real_t x39 = x32 - x34 + x35 + x37;
    const real_t x40 = c[10] * x17;
    const real_t x41 = c[4] * x17;
    const real_t x42 = c[7] * x17;
    const real_t x43 = c[1] * x17;
    const real_t x44 = c[11] * x17;
    const real_t x45 = c[5] * x17;
    const real_t x46 = c[8] * x17;
    const real_t x47 = c[2] * x17;
    output[0] = x18 * x7 + x19 * x20 + x22 * x23 + x24 * x25;
    output[1] = x18 * x26 + x20 * x29 + x23 * x30 + x25 * x31;
    output[2] = x18 * x33 + x20 * x36 + x23 * x38 + x25 * x39;
    output[3] = x19 * x42 + x22 * x40 + x24 * x43 + x41 * x7;
    output[4] = x26 * x41 + x29 * x42 + x30 * x40 + x31 * x43;
    output[5] = x33 * x41 + x36 * x42 + x38 * x40 + x39 * x43;
    output[6] = x19 * x46 + x22 * x44 + x24 * x47 + x45 * x7;
    output[7] = x26 * x45 + x29 * x46 + x30 * x44 + x31 * x47;
    output[8] = x33 * x45 + x36 * x46 + x38 * x44 + x39 * x47;
}

//
static SFEM_INLINE void tet4_fe_grad(
    // x
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // z
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3,
    // Coefficients
    const real_t *c,

    // Out
    real_t *output) {
    // FLOATING POINT OPS!
    //  - Result: 9*ADD + 3*ASSIGNMENT + 20*MUL
    //  - Subexpressions: 2*ADD + DIV + 28*MUL + 18*SUB
    const real_t x0 = py0 - py2;
    const real_t x1 = pz0 - pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = py0 - py3;
    const real_t x4 = pz0 - pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = px0 - px1;
    const real_t x8 = px0 - px2;
    const real_t x9 = pz0 - pz1;
    const real_t x10 = x3 * x9;
    const real_t x11 = px0 - px3;
    const real_t x12 = py0 - py1;
    const real_t x13 = x12 * x4;
    const real_t x14 = x1 * x12;
    const real_t x15 = x0 * x9;
    const real_t x16 = 1.0 / (x10 * x8 + x11 * x13 - x11 * x15 - x14 * x8 + x2 * x7 - x5 * x7);
    const real_t x17 = c[1] * x16;
    const real_t x18 = c[2] * x16;
    const real_t x19 = x13 - x15;
    const real_t x20 = c[3] * x16;
    const real_t x21 = c[0] * x16;
    const real_t x22 = x1 * x8 - x11 * x4;
    const real_t x23 = x1 * x7;
    const real_t x24 = x11 * x9;
    const real_t x25 = x4 * x7 - x8 * x9;
    const real_t x26 = -x0 * x11 + x3 * x8;
    const real_t x27 = x3 * x7;
    const real_t x28 = x11 * x12;
    const real_t x29 = x0 * x7 - x12 * x8;
    output[0] = -x17 * x6 + x18 * (-x10 + x14) - x19 * x20 + x21 * (x10 - x14 + x19 + x6);
    output[1] = x17 * x22 + x18 * (-x23 + x24) + x20 * x25 + x21 * (-x22 + x23 - x24 - x25);
    output[2] = -x17 * x26 + x18 * (x27 - x28) - x20 * x29 + x21 * (x26 - x27 + x28 + x29);
}

//
// static SFEM_INLINE void tet4_fe_fun(
//     // Coefficients
//     const real_t *c,
//     const real_t qx,
//     const real_t qy,
//     const real_t qz,
//     // Out
//     real_t *output) {
//     // FLOATING POINT OPS!
//     //  - Result: 2*ADD + ASSIGNMENT + 7*MUL
//     //  - Subexpressions: 0
//     output[0] = c[0] * (-qx - qy - qz + 1) + c[1] * qx + c[2] * qy + c[3] * qz;
// }

static SFEM_INLINE void tet4_shape_fun(const real_t qx, const real_t qy, const real_t qz, real_t *f) {
    f[0] = 1 - qx - qy - qz;
    f[1] = qx;
    f[2] = qy;
    f[3] = qz;
}

static SFEM_INLINE void tet4_shape_grad(
    // x
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    // y
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    // z
    const real_t pz0,
    const real_t pz1,
    const real_t pz2,
    const real_t pz3,
    real_t *g) {
    // FLOATING POINT OPS!
    //   - Result: 6*ADD + 12*ASSIGNMENT + 20*MUL
    //   - Subexpressions: 2*ADD + DIV + 24*MUL + 18*SUB
    const real_t x0 = py0 - py3;
    const real_t x1 = pz0 - pz1;
    const real_t x2 = x0 * x1;
    const real_t x3 = py0 - py1;
    const real_t x4 = pz0 - pz3;
    const real_t x5 = x3 * x4;
    const real_t x6 = py0 - py2;
    const real_t x7 = x4 * x6;
    const real_t x8 = pz0 - pz2;
    const real_t x9 = x0 * x8;
    const real_t x10 = x7 - x9;
    const real_t x11 = x3 * x8;
    const real_t x12 = x1 * x6;
    const real_t x13 = x11 - x12;
    const real_t x14 = px0 - px1;
    const real_t x15 = px0 - px2;
    const real_t x16 = px0 - px3;
    const real_t x17 = 1.0 / (x11 * x16 - x12 * x16 + x14 * x7 - x14 * x9 + x15 * x2 - x15 * x5);
    const real_t x18 = x1 * x16;
    const real_t x19 = x14 * x4;
    const real_t x20 = x15 * x4 - x16 * x8;
    const real_t x21 = -x1 * x15 + x14 * x8;
    const real_t x22 = x16 * x3;
    const real_t x23 = x0 * x14;
    const real_t x24 = x0 * x15 - x16 * x6;
    const real_t x25 = x14 * x6 - x15 * x3;
    g[0] = x17 * (x10 + x13 + x2 - x5);
    g[1] = x17 * (-x18 + x19 - x20 - x21);
    g[2] = x17 * (x22 - x23 + x24 + x25);
    g[3] = -x10 * x17;
    g[4] = x17 * x20;
    g[5] = -x17 * x24;
    g[6] = x17 * (-x2 + x5);
    g[7] = x17 * (x18 - x19);
    g[8] = x17 * (-x22 + x23);
    g[9] = -x13 * x17;
    g[10] = x17 * x21;
    g[11] = -x17 * x25;
}

static SFEM_INLINE void isotropic_phasefield_AT2_energy(const real_t mu,
                                                        const real_t lambda,
                                                        const real_t Gc,
                                                        const real_t ls,
                                                        const real_t c,
                                                        const real_t *gradc,
                                                        const real_t *gradu,
                                                        const real_t dV,
                                                        real_t *element_scalar) {
    // FLOATING POINT OPS!
    //      - Result: 10*ADD + ADDAUGMENTEDASSIGNMENT + 17*MUL + 13*POW
    //      - Subexpressions: 0
    element_scalar[0] +=
        dV * ((1.0 / 2.0) * Gc * (pow(c, 2) / ls + ls * (pow(gradc[0], 2) + pow(gradc[1], 2) + pow(gradc[2], 2))) +
              pow(1 - c, 2) * ((1.0 / 2.0) * lambda * pow(gradu[0] + gradu[4] + gradu[8], 2) +
                               mu * (pow(gradu[0], 2) + pow(gradu[4], 2) + pow(gradu[8], 2) +
                                     2 * pow((1.0 / 2.0) * gradu[1] + (1.0 / 2.0) * gradu[3], 2) +
                                     2 * pow((1.0 / 2.0) * gradu[2] + (1.0 / 2.0) * gradu[6], 2) +
                                     2 * pow((1.0 / 2.0) * gradu[5] + (1.0 / 2.0) * gradu[7], 2))));
}

static SFEM_INLINE void isotropic_phasefield_AT2_gradient(const real_t mu,
                                                          const real_t lambda,
                                                          const real_t Gc,
                                                          const real_t ls,
                                                          const real_t c,
                                                          const real_t *gradc,
                                                          const real_t *gradu,
                                                          const real_t test,
                                                          const real_t *test_grad,
                                                          const real_t dV,
                                                          real_t *element_vector) {
    // FLOATING POINT OPS!
    //      - Result: 15*ADD + 4*ADDAUGMENTEDASSIGNMENT + 34*MUL + 8*POW
    //      - Subexpressions: 5*ADD + DIV + 10*MUL + POW + SUB
    const real_t x0 = pow(1 - c, 2);
    const real_t x1 = test_grad[1] * x0;
    const real_t x2 = mu * (gradu[1] + gradu[3]);
    const real_t x3 = gradu[2] + gradu[6];
    const real_t x4 = test_grad[2] * x0;
    const real_t x5 = mu * x4;
    const real_t x6 = 2 * gradu[0];
    const real_t x7 = 2 * gradu[4];
    const real_t x8 = 2 * gradu[8];
    const real_t x9 = (1.0 / 2.0) * lambda;
    const real_t x10 = x9 * (x6 + x7 + x8);
    const real_t x11 = test_grad[0] * x0;
    const real_t x12 = gradu[5] + gradu[7];
    const real_t x13 = Gc * ls;
    element_vector[0] += dV * (x1 * x2 + x11 * (mu * x6 + x10) + x3 * x5);
    element_vector[1] += dV * (x1 * (mu * x7 + x10) + x11 * x2 + x12 * x5);
    element_vector[2] += dV * (mu * x1 * x12 + mu * x11 * x3 + x4 * (mu * x8 + x10));
    element_vector[3] +=
        dV * (gradc[0] * test_grad[0] * x13 + gradc[1] * test_grad[1] * x13 + gradc[2] * test_grad[2] * x13 +
              test * (Gc * c / ls + (2 * c - 2) * (mu * (pow(gradu[0], 2) + pow(gradu[4], 2) + pow(gradu[8], 2) +
                                                         2 * pow((1.0 / 2.0) * gradu[1] + (1.0 / 2.0) * gradu[3], 2) +
                                                         2 * pow((1.0 / 2.0) * gradu[2] + (1.0 / 2.0) * gradu[6], 2) +
                                                         2 * pow((1.0 / 2.0) * gradu[5] + (1.0 / 2.0) * gradu[7], 2)) +
                                                   x9 * pow(gradu[0] + gradu[4] + gradu[8], 2))));
}

static SFEM_INLINE void isotropic_phasefield_AT2_hessian(const real_t mu,
                                                         const real_t lambda,
                                                         const real_t Gc,
                                                         const real_t ls,
                                                         const real_t c,
                                                         const real_t *gradu,
                                                         const real_t test,
                                                         const real_t trial,
                                                         const real_t *test_grad,
                                                         const real_t *trial_grad,
                                                         const real_t dV,
                                                         real_t *element_matrix) {
    // FLOATING POINT OPS!
    //      - Result: 16*ADD + 16*ADDAUGMENTEDASSIGNMENT + 41*MUL + 8*POW
    //      - Subexpressions: 15*ADD + DIV + 46*MUL + POW + 2*SUB
    const real_t x0 = pow(1 - c, 2);
    const real_t x1 = mu * x0;
    const real_t x2 = test_grad[1] * trial_grad[1];
    const real_t x3 = x1 * x2;
    const real_t x4 = trial_grad[2] * x1;
    const real_t x5 = test_grad[2] * x4;
    const real_t x6 = 2 * mu;
    const real_t x7 = lambda + x6;
    const real_t x8 = test_grad[0] * x0;
    const real_t x9 = trial_grad[0] * x8;
    const real_t x10 = test_grad[1] * trial_grad[0];
    const real_t x11 = lambda * x0;
    const real_t x12 = mu * x8;
    const real_t x13 = test_grad[2] * x11;
    const real_t x14 = 2 * c - 2;
    const real_t x15 = trial_grad[1] * x14;
    const real_t x16 = mu * (gradu[1] + gradu[3]);
    const real_t x17 = gradu[2] + gradu[6];
    const real_t x18 = trial_grad[2] * x14;
    const real_t x19 = mu * x18;
    const real_t x20 = (1.0 / 2.0) * lambda * (2 * gradu[0] + 2 * gradu[4] + 2 * gradu[8]);
    const real_t x21 = trial_grad[0] * x14;
    const real_t x22 = dV * test;
    const real_t x23 = x22 * (x15 * x16 + x17 * x19 + x21 * (gradu[0] * x6 + x20));
    const real_t x24 = lambda * x8;
    const real_t x25 = mu * x9;
    const real_t x26 = x0 * x7;
    const real_t x27 = gradu[5] + gradu[7];
    const real_t x28 = x22 * (x15 * (gradu[4] * x6 + x20) + x16 * x21 + x19 * x27);
    const real_t x29 = test_grad[2] * x1;
    const real_t x30 = test_grad[2] * trial_grad[2];
    const real_t x31 = x22 * (mu * x15 * x27 + mu * x17 * x21 + x18 * (gradu[8] * x6 + x20));
    const real_t x32 = Gc * ls;
    element_matrix[0] += dV * (x3 + x5 + x7 * x9);
    element_matrix[1] += dV * (trial_grad[1] * x12 + x10 * x11);
    element_matrix[2] += dV * (trial_grad[0] * x13 + trial_grad[2] * x12);
    element_matrix[3] += x23;
    element_matrix[4] += dV * (trial_grad[1] * x24 + x1 * x10);
    element_matrix[5] += dV * (x2 * x26 + x25 + x5);
    element_matrix[6] += dV * (test_grad[1] * x4 + trial_grad[1] * x13);
    element_matrix[7] += x28;
    element_matrix[8] += dV * (trial_grad[0] * x29 + trial_grad[2] * x24);
    element_matrix[9] += dV * (test_grad[1] * trial_grad[2] * x11 + trial_grad[1] * x29);
    element_matrix[10] += dV * (x25 + x26 * x30 + x3);
    element_matrix[11] += x31;
    element_matrix[12] += x23;
    element_matrix[13] += x28;
    element_matrix[14] += x31;
    element_matrix[15] += dV * (test * trial *
                                    (Gc / ls + lambda * pow(gradu[0] + gradu[4] + gradu[8], 2) +
                                     x6 * (pow(gradu[0], 2) + pow(gradu[4], 2) + pow(gradu[8], 2) +
                                           2 * pow((1.0 / 2.0) * gradu[1] + (1.0 / 2.0) * gradu[3], 2) +
                                           2 * pow((1.0 / 2.0) * gradu[2] + (1.0 / 2.0) * gradu[6], 2) +
                                           2 * pow((1.0 / 2.0) * gradu[5] + (1.0 / 2.0) * gradu[7], 2))) +
                                test_grad[0] * trial_grad[0] * x32 + x2 * x32 + x30 * x32);
}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
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

static SFEM_INLINE void find_cols4(const idx_t *targets, const idx_t *const row, const int lenrow, int *ks) {
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

#ifndef NDEBUG
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
#endif

#ifndef NDEBUG
static void numerate(int n, real_t *const element_matrix) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            element_matrix[i * n + j] = i * n + j;
        }
    }
}
#endif

static const int n_qp = 8;
static const real_t qx[8] = {0.0, 1.0, 0.0, 0.0, 0.333333333333, 0.333333333333, 0.0, 0.333333333333};
static const real_t qy[8] = {0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.0, 0.333333333333, 0.333333333333};
static const real_t qz[8] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.333333333333, 0.333333333333, 0.333333333333};
static const real_t qw[8] = {0.025, 0.025, 0.025, 0.025, 0.225, 0.225, 0.225, 0.225};

void isotropic_phasefield_for_fracture_assemble_hessian(const ptrdiff_t nelements,
                                                        const ptrdiff_t nnodes,
                                                        idx_t *const elems[4],
                                                        geom_t *const xyz[3],
                                                        const real_t mu,
                                                        const real_t lambda,
                                                        const real_t Gc,
                                                        const real_t ls,
                                                        const real_t *const u,
                                                        count_t *const rowptr,
                                                        idx_t *const colidx,
                                                        real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int block_size = 4;
    static const int mat_block_size = block_size * block_size;

    idx_t ev[4];
    idx_t ks[4];

    real_t element_node_matrix[(4 * 4)];
    real_t element_displacement[(4 * 3)];
    real_t element_phasefield[4];

    real_t shape_fun[4];
    real_t shape_grad[4 * 3];
    real_t grad_displacement[3 * 3];

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
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
            const idx_t dof = ev[enode] * block_size;

            for (int b = 0; b < 3; ++b) {
                element_displacement[enode * 3 + b] = u[dof + b];
            }

            element_phasefield[enode] = u[dof + 3];
        }

        const real_t measure =
            tet4_measure(x[i0], x[i1], x[i2], x[i3], y[i0], y[i1], y[i2], y[i3], z[i0], z[i1], z[i2], z[i3]);

        // Gradient does not depend on qp for tet4
        tet4_shape_grad(
            //
            x[i0],
            x[i1],
            x[i2],
            x[i3],
            //
            y[i0],
            y[i1],
            y[i2],
            y[i3],
            //
            z[i0],
            z[i1],
            z[i2],
            z[i3],
            shape_grad);

        tet4_fe_tgrad(x[i0],
                      x[i1],
                      x[i2],
                      x[i3],
                      y[i0],
                      y[i1],
                      y[i2],
                      y[i3],
                      z[i0],
                      z[i1],
                      z[i2],
                      z[i3],
                      element_displacement,
                      grad_displacement);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
            const idx_t *row = &colidx[rowptr[dof_i]];
            find_cols4(ev, row, lenrow, ks);

            real_t *row_blocks = &values[rowptr[dof_i] * mat_block_size];

            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                memset(element_node_matrix, 0, sizeof(real_t) * (4 * 4));

                for (int q = 0; q < n_qp; ++q) {
                    const real_t dV = qw[q] * measure;

                    tet4_shape_fun(qx[q], qy[q], qz[q], shape_fun);
                    const real_t c = shape_fun[0] * element_phasefield[0] + shape_fun[1] * element_phasefield[1] +
                                     shape_fun[2] * element_phasefield[2] + shape_fun[3] * element_phasefield[3];

                    isotropic_phasefield_AT2_hessian(mu,
                                                     lambda,
                                                     Gc,
                                                     ls,
                                                     c,
                                                     grad_displacement,
                                                     shape_fun[edof_i],
                                                     shape_fun[edof_j],
                                                     &shape_grad[edof_i * 3],
                                                     &shape_grad[edof_j * 3],
                                                     dV,
                                                     element_node_matrix);
                }

                // printf("%d, %d)\n", edof_i, edof_j);
                // for (int bj = 0; bj < block_size; ++bj) {
                //     for (int bi = 0; bi < block_size; ++bi) {
                //         printf("%g ", element_node_matrix[bi * block_size + bj]);
                //     }
                //     printf("\n");
                // }

                // printf("-----------------------\n");

                const idx_t block_k = ks[edof_j] * mat_block_size;
                real_t *block = &row_blocks[block_k];

                // Iterate over dimensions
                for (int bj = 0; bj < block_size; ++bj) {
                    const idx_t offset_j = bj * block_size;

                    for (int bi = 0; bi < block_size; ++bi) {
                        const real_t val = element_node_matrix[bi * block_size + bj];

                        assert(val == val);

                        block[offset_j + bi] += val;
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("isotropic_phasefield_for_fracture.c: assemble_hessian\t%g seconds\n", tock - tick);
}

void isotropic_phasefield_for_fracture_assemble_gradient(const ptrdiff_t nelements,
                                                         const ptrdiff_t nnodes,
                                                         idx_t *const elems[4],
                                                         geom_t *const xyz[3],
                                                         const real_t mu,
                                                         const real_t lambda,
                                                         const real_t Gc,
                                                         const real_t ls,
                                                         const real_t *const u,
                                                         real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int block_size = 4;

    idx_t ev[4];
    real_t element_node_vector[4];
    real_t element_displacement[4 * 3];
    real_t element_phasefield[4];

    real_t shape_fun[4];
    real_t shape_grad[4 * 3];

    real_t grad_phasefield[3];
    real_t grad_displacement[3 * 3];

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
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

            for (int b = 0; b < 3; ++b) {
                element_displacement[enode * 3 + b] = u[dof + b];
            }

            element_phasefield[enode] = u[dof + 3];
        }

        const real_t measure =
            tet4_measure(x[i0], x[i1], x[i2], x[i3], y[i0], y[i1], y[i2], y[i3], z[i0], z[i1], z[i2], z[i3]);

        // Gradient does not depend on qp for tet4
        tet4_shape_grad(x[i0], x[i1], x[i2], x[i3], y[i0], y[i1], y[i2], y[i3], z[i0], z[i1], z[i2], z[i3], shape_grad);

        tet4_fe_tgrad(x[i0],
                      x[i1],
                      x[i2],
                      x[i3],
                      y[i0],
                      y[i1],
                      y[i2],
                      y[i3],
                      z[i0],
                      z[i1],
                      z[i2],
                      z[i3],
                      element_displacement,
                      grad_displacement);

        tet4_fe_grad(x[i0],
                     x[i1],
                     x[i2],
                     x[i3],
                     y[i0],
                     y[i1],
                     y[i2],
                     y[i3],
                     z[i0],
                     z[i1],
                     z[i2],
                     z[i3],
                     element_phasefield,
                     grad_phasefield);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i] * block_size;

            memset(element_node_vector, 0, sizeof(real_t) * (4));

            for (int q = 0; q < n_qp; ++q) {
                const real_t dV = qw[q] * measure;

                tet4_shape_fun(qx[q], qy[q], qz[q], shape_fun);
                const real_t c = shape_fun[0] * element_phasefield[0] + shape_fun[1] * element_phasefield[1] +
                                 shape_fun[2] * element_phasefield[2] + shape_fun[3] * element_phasefield[3];

                isotropic_phasefield_AT2_gradient(mu,
                                                  lambda,
                                                  Gc,
                                                  ls,
                                                  c,
                                                  grad_phasefield,
                                                  grad_displacement,
                                                  shape_fun[edof_i],
                                                  &shape_grad[edof_i * 3],
                                                  dV,
                                                  element_node_vector);
            }

            for (int bi = 0; bi < block_size; ++bi) {
                values[dof_i + bi] += element_node_vector[bi];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("isotropic_phasefield_for_fracture.c: assemble_gradient\t%g seconds\n", tock - tick);
}

void isotropic_phasefield_for_fracture_assemble_value(const ptrdiff_t nelements,
                                                      const ptrdiff_t nnodes,
                                                      idx_t *const elems[4],
                                                      geom_t *const xyz[3],
                                                      const real_t mu,
                                                      const real_t lambda,
                                                      const real_t Gc,
                                                      const real_t ls,
                                                      const real_t *const u,
                                                      real_t *const value) {
    SFEM_UNUSED(nnodes);
    
    double tick = MPI_Wtime();

    static const int block_size = 4;
    // static const int mat_block_size = block_size * block_size;

    idx_t ev[4];
    // real_t element_node_vector[4];
    real_t element_displacement[4 * 3];
    real_t element_phasefield[4];

    real_t shape_fun[4];
    // real_t shape_grad[4 * 3];

    real_t grad_phasefield[3];
    real_t grad_displacement[3 * 3];

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];
    const geom_t *const z = xyz[2];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
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

            for (int b = 0; b < 3; ++b) {
                element_displacement[enode * 3 + b] = u[dof + b];
            }

            element_phasefield[enode] = u[dof + 3];
        }

        const real_t measure =
            tet4_measure(x[i0], x[i1], x[i2], x[i3], y[i0], y[i1], y[i2], y[i3], z[i0], z[i1], z[i2], z[i3]);

        tet4_fe_tgrad(x[i0],
                      x[i1],
                      x[i2],
                      x[i3],
                      y[i0],
                      y[i1],
                      y[i2],
                      y[i3],
                      z[i0],
                      z[i1],
                      z[i2],
                      z[i3],
                      element_displacement,
                      grad_displacement);

        tet4_fe_grad(x[i0],
                     x[i1],
                     x[i2],
                     x[i3],
                     y[i0],
                     y[i1],
                     y[i2],
                     y[i3],
                     z[i0],
                     z[i1],
                     z[i2],
                     z[i3],
                     element_phasefield,
                     grad_phasefield);

        real_t element_scalar = 0;

        for (int q = 0; q < n_qp; ++q) {
            const real_t dV = qw[q] * measure;

            tet4_shape_fun(qx[q], qy[q], qz[q], shape_fun);
            const real_t c = shape_fun[0] * element_phasefield[0] + shape_fun[1] * element_phasefield[1] +
                             shape_fun[2] * element_phasefield[2] + shape_fun[3] * element_phasefield[3];

            isotropic_phasefield_AT2_energy(
                mu, lambda, Gc, ls, c, grad_phasefield, grad_displacement, dV, &element_scalar);
        }

        *value += element_scalar;
    }

    double tock = MPI_Wtime();
    printf("isotropic_phasefield_for_fracture.c: assemble_value\t%g seconds\n", tock - tick);
}
